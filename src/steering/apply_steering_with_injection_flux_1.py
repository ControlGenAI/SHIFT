"""
apply_steering_dual_branch.py

Your existing code steers output[1][0] = TEXT branch only.
This adds output[0][0] = IMAGE branch steering in the same hook.

Vector format (dual-stream):
  vector[step][f"layer_{idx}"] = {"img": tensor, "txt": tensor}

Falls back to original format if no "img"/"txt" keys found:
  vector[step][f"layer_{idx}"] = tensor  (treated as text-only, same as before)

Text encoder steering (T5 sequence + CLIP pooled) is unchanged — 
separate file via --steer_txt.
"""

import torch
import numpy as np
import os
import torch.nn.functional as F
import argparse
import sklearn.svm._classes
from typing import Dict, List, Tuple, Union, Optional
from datasets import load_dataset
import sys
from pathlib import Path
STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))
from src.models.sd_3 import StableDiffusion3Pipeline
from src.models.flux import FluxPipeline
from src.utils.utils import calculate_cls_score


# ==============================================================================
# Unchanged from your original code
# ==============================================================================

def vector_is_dense_per_token(vector_type: str) -> bool:
    """True for mean-diff and OT steering tensors (per-token, like diff)."""
    return vector_type == "diff" or vector_type.startswith("ot_") or "ot" in vector_type


def load_steering_data(
    svm_model_path: Optional[str], 
    scores_path: Optional[str], 
    token_best_path: Optional[str], 
    quantile_level: float
) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray]]:
    """Loads SVM models, best tokens, and scores using original logic."""
    
    models = None
    if svm_model_path and os.path.exists(svm_model_path):
        with torch.serialization.safe_globals([sklearn.svm._classes.SVC]):
            models = torch.load(svm_model_path, weights_only=False)

    tokens_best = None
    if token_best_path and os.path.exists(token_best_path):
        tokens_best = torch.load(token_best_path, weights_only=False)
     
    scores_all = None
    if scores_path and os.path.exists(scores_path):
        # Handle both tensor and numpy loads safely
        data = torch.load(scores_path, weights_only=False)
        scores_all = data if torch.is_tensor(data) else torch.from_numpy(data)

    quantiles = None
    if scores_all is not None and scores_path:
        scores_np = scores_all.numpy()
        if 'block' in scores_path:
            quantiles = np.quantile(scores_np, q=quantile_level, axis=2)
        elif 'timestep' in scores_path:
            quantiles = np.quantile(scores_np, q=quantile_level, axis=(1, 2))
        else:
            quantiles = np.quantile(scores_np, q=quantile_level)
    return models, tokens_best, scores_all, quantiles


class SteeringEngine:
    @classmethod
    def apply_steering(
        cls, 
        activations: torch.Tensor, 
        steering_vec: torch.Tensor, 
        args, 
        score_val: torch.Tensor
    ) -> torch.Tensor:
        
        def calculate_sim(act_unit, v_unit, args):
            sim = F.cosine_similarity(act_unit, v_unit, dim=-1)
            k_val = max(1, int(sim.shape[0] * args.top_k_percent))
            threshold = torch.topk(sim.flatten(), k_val).values[-1]
            mask = (sim >= threshold).float().unsqueeze(-1)
            
            # Use similarity weight only for removal tasks
            weight = sim.unsqueeze(-1)
            return mask, weight.clip(0, 2)
        
        dtype = activations.dtype
        act_f32 = activations.float()
        orig_norm = torch.norm(act_f32, dim=-1, keepdim=True) + 1e-6
        act_unit = act_f32 / orig_norm
        
        v_unit = steering_vec.float() / (torch.norm(steering_vec.float(), dim=-1, keepdim=True) + 1e-6)
        v_steer = v_unit

        if args.use_ssim_mask:
            if vector_is_dense_per_token(args.vector_type) or args.steering_type == 'mean':
                mask, weight = calculate_sim(act_unit, v_unit, args)
            else:
                masks = []
                weights = []
                for i, v in enumerate(v_unit):
                    mask, weight = calculate_sim(act_unit, v_unit[i], args)
                    masks.append(mask)
                    weights.append(weight)
                
                mask = torch.stack(masks)
                weight = torch.stack(weights)
                

        else:
            mask, weight = 1.0, 1.0

        # 3. Task Logic
        if args.task == 'remove':
            if vector_is_dense_per_token(args.vector_type) or args.steering_type == 'mean':
                adjustment = args.strength * weight * v_steer.to(activations.dtype) * mask * score_val
            else:
                score_val = score_val.unsqueeze(1)
                adjustment = args.strength * weight * v_steer.to(activations.dtype) * mask * score_val
                adjustment = adjustment.mean(0)
            steered = activations - (adjustment)
        else:
            # Masked addition targets concept-relevant tokens
            if vector_is_dense_per_token(args.vector_type) or args.steering_type == 'mean':
                print(f"v_steer shape: {v_steer.shape}", activations.shape)
                print(f"score_val shape: {score_val.shape}", args.strength)
                adjustment = args.strength * v_steer.to(activations.dtype) * score_val  
            else:
                assert False
                score_val = score_val.unsqueeze(1)
                adjustment = args.strength * v_steer.to(activations.dtype).unsqueeze(1) * mask * score_val.unsqueeze(1)
                adjustment = adjustment.mean(0)
            steered = activations + (adjustment)

        # 4. Energy Restoration
        steered_unit = steered.float() / (torch.norm(steered.float(), dim=-1, keepdim=True) + 1e-6)
        return (steered_unit * orig_norm).to(dtype)


MODEL_CONFIGS = {
    "flux": {"last_layer": 56, "out_idx": 1, "inner_idx": 0, "dtype": torch.bfloat16},
    "sd3":  {"last_layer": 23, "out_idx": 1, "inner_idx": 1, "dtype": torch.float16},
}


# ==============================================================================
# Helper: extract img/txt vectors from dual-stream format
# ==============================================================================

def split_vector(vector: dict) -> Tuple[dict, dict]:
    """
    If vector is dual-stream {step: {layer: {"img": t, "txt": t}}},
    split into two separate dicts.
    
    If vector is legacy {step: {layer: tensor}}, treat as txt-only (original behavior).
    
    Returns: (img_vector, txt_vector)
      - img_vector: {step: {layer: tensor}} or empty dict
      - txt_vector: {step: {layer: tensor}} or original vector
    """
    img_vec = {}
    txt_vec = {}
    
    # Check if dual-stream
    is_dual = False
    for step in vector:
        if not isinstance(step, int):
            continue
        for layer_key in vector[step]:
            val = vector[step][layer_key]
            if isinstance(val, dict) and ("img" in val or "txt" in val):
                is_dual = True
            break
        break
    
    if not is_dual:
        # Legacy format: everything is the text branch (your original behavior)
        return {}, vector
    
    for step in vector:
        if not isinstance(step, int):
            continue
        img_vec[step] = {}
        txt_vec[step] = {}
        for layer_key in vector[step]:
            val = vector[step][layer_key]
            if isinstance(val, dict):
                if "img" in val:
                    t = val["img"]
                    while t.dim() > 2:
                        t = t.squeeze(0)
                    img_vec[step][layer_key] = t
                if "txt" in val:
                    t = val["txt"]
                    while t.dim() > 2:
                        t = t.squeeze(0)
                    txt_vec[step][layer_key] = t
            else:
                # Mixed format — treat plain tensors as txt
                txt_vec[step][layer_key] = val
    
    return img_vec, txt_vec


# ==============================================================================
# DUAL BRANCH STEERING HOOK
# ==============================================================================

def apply_attention_steering(pipe, args, vector, idx=None):
    """
    Hooks modules to steer BOTH image and text branches.
    
    Supported injection points (matches `get_vector_1.py` extraction points):
      - attn: hook `block.attn` output (tuple (img_attn_out, txt_attn_out))
      - block: hook whole `FluxTransformerBlock` output (tuple (txt, img))
      - ff: hook `block.ff` (img) and `block.ff_context` (txt) outputs (single tensors)
    
    Note: for FLUX we rely on `state["step"]` provided by the pipeline callback
    to align step indices with extracted activations.
    """
    m_key = "flux" if "flux" in args.model_name.lower() else "sd3"
    cfg = MODEL_CONFIGS[m_key]
    state = {"step": 0}
    hook_handles = []

    # Split into img and txt vectors
    img_vector, txt_vector = split_vector(vector)
    
    has_img = bool(img_vector)
    has_txt = bool(txt_vector)
    
    print(f"  Steering branches: img={'YES' if has_img else 'NO'}, txt={'YES' if has_txt else 'NO'}")
    if has_img:
        # Print shape of first available vector
        for s in img_vector:
            for l in img_vector[s]:
                print(f"  img vector shape: {tuple(img_vector[s][l].shape)}")
                break
            break
    if has_txt:
        for s in txt_vector:
            for l in txt_vector[s]:
                print(f"  txt vector shape: {tuple(txt_vector[s][l].shape)}")
                break
            break

    # Load auxiliary data
    eigen_path = os.path.join(args.data_dir, "cos_sep.pt")
    svm_path = os.path.join(args.data_dir, "base_0.85_20_svm_models.pt")
    scr_path = os.path.join(args.data_dir, "base_0.85_20_scores.pt")
    tok_path = os.path.join(args.data_dir, "base_0.85_20_tokens.pt")

    models, tokens_best, scores_all, quantiles = load_steering_data(
        svm_path, scr_path, tok_path, args.quantile_level
    )
   
    try:
        eigen_info = torch.load(eigen_path, weights_only=False)
    except:
        eigen_info = None

    def _get_score_val(step, layer_idx, to_modify, models, scores_all, eigen_info, cfg, args):
        """Compute classifier score — same logic as your original."""
        score_val = torch.ones((1, 1), device=to_modify.device, dtype=to_modify.dtype)
        current_signal = 1.0
        if scores_all is not None:
            layer_scores = scores_all[:, step, layer_idx]
            current_signal = layer_scores.float()

        if args.use_cls and models:
            ensemble = models.get(step, {}).get(f"layer_{layer_idx}")
            if ensemble:
                mean_act = to_modify.mean(0, keepdim=True)
                mean_act = mean_act / (mean_act.norm(dim=-1, keepdim=True) + 1e-6)
                votes = [
                    calculate_cls_score(
                        mean_act.cpu().float(), args.cls_min, m,
                        args.cls_type, task=args.task, use_distance=False
                    )[0]
                    if current_signal[i] > args.min_signal_threshold else 1.0
                    for i, m in enumerate(ensemble)
                ]
                score_val = torch.tensor(votes).to(to_modify.device, to_modify.dtype)
                if vector_is_dense_per_token(args.vector_type) or args.steering_type == 'mean':
                    score_val = torch.mean(score_val, dim=0, keepdim=True)

        return score_val

    def _apply_to_branch(to_modify, sv_raw, args, score_val, strength):
        """Apply steering to one branch with given strength."""
        if args.steering_type == 'mean':
            steering_vec = sv_raw.mean(0, keepdim=True).to(to_modify.device)
        else:
            steering_vec = sv_raw.to(to_modify.device, to_modify.dtype)

        #print(f"steering_vec shape: {steering_vec.shape}")
        # Temporarily override strength
        orig_strength = args.strength
        args.strength = strength
        result = SteeringEngine.apply_steering(to_modify, steering_vec, args, score_val)
        args.strength = orig_strength
        return result

    def _get_layer_vectors(step: int, layer_idx: int):
        layer_key = f"layer_{layer_idx}"
        has_txt_vec = (step in txt_vector and layer_key in txt_vector.get(step, {}))
        has_img_vec = (step in img_vector and layer_key in img_vector.get(step, {}))
        sv_txt = txt_vector[step][layer_key] if has_txt_vec else None
        sv_img = img_vector[step][layer_key] if has_img_vec else None
        return layer_key, sv_img, sv_txt

    def steering_hook(layer_idx: int):
        def hook(module, input, output):
            step = state["step"]
            # Skip conditions
            if args.block_steering != 'all' and layer_idx not in args.block_steering:
                return output
            if args.t_steering != 'all' and step not in args.t_steering:
                return output
            
            print(f"step: {step}, layer_idx: {layer_idx}")
            layer_key, sv_img, sv_txt = _get_layer_vectors(step, layer_idx)
            if sv_img is None and sv_txt is None:
                return output

            # ----------------------------
            # injection_point = "attn"
            #   output = (img_attn_out, txt_attn_out) OR nested
            # ----------------------------
            if args.injection_point == "attn":
                if not (isinstance(output, tuple) and len(output) == 2):
                    return output

                act_tuple = list(output)

                # TEXT BRANCH (tuple[1])
                if sv_txt is not None:
                    txt_hidden = act_tuple[1]
                    try:
                        txt_to_modify = txt_hidden[cfg["inner_idx"]].clone()
                        use_inner_txt = True
                    except (IndexError, TypeError):
                        txt_to_modify = txt_hidden.clone()
                        use_inner_txt = False

                    score_val = _get_score_val(step, layer_idx, txt_to_modify, models, scores_all, eigen_info, cfg, args)
                    steered_txt = _apply_to_branch(txt_to_modify, sv_txt, args, score_val, args.strength)

                    if use_inner_txt:
                        act_tuple[1][cfg["inner_idx"]] = steered_txt
                    else:
                        act_tuple[1] = steered_txt

                # IMAGE BRANCH (tuple[0])
                if sv_img is not None:
                    img_hidden = act_tuple[0]
                    try:
                        img_to_modify = img_hidden[cfg["inner_idx"]].clone()
                        use_inner_img = True
                    except (IndexError, TypeError):
                        img_to_modify = img_hidden.clone()
                        use_inner_img = False

                    score_val = _get_score_val(step, layer_idx, img_to_modify, models, scores_all, eigen_info, cfg, args)
                    steered_img = _apply_to_branch(img_to_modify, sv_img, args, score_val, args.strength_img)

                    if use_inner_img:
                        act_tuple[0][cfg["inner_idx"]] = steered_img
                    else:
                        act_tuple[0] = steered_img

                return tuple(act_tuple)

            # ----------------------------
            # injection_point = "block"
            #   output = (txt, img)
            # ----------------------------
            if args.injection_point == "block":
                if not (isinstance(output, tuple) and len(output) == 2):
                    return output

                txt_hidden, img_hidden = output[0], output[1]
                txt_new, img_new = txt_hidden, img_hidden

                if sv_txt is not None:
                    txt_to_modify = txt_hidden.clone()
                    score_val = _get_score_val(step, layer_idx, txt_to_modify, models, scores_all, eigen_info, cfg, args)
                    txt_new = _apply_to_branch(txt_to_modify, sv_txt, args, score_val, args.strength)

                if sv_img is not None:
                    img_to_modify = img_hidden.clone()
                    score_val = _get_score_val(step, layer_idx, img_to_modify, models, scores_all, eigen_info, cfg, args)
                    img_new = _apply_to_branch(img_to_modify, sv_img, args, score_val, args.strength_img)

                return (txt_new, img_new)

            return output
        return hook

    def steering_hook_ff(layer_idx: int, branch: str):
        """
        injection_point = "ff"
          - branch="img": hook block.ff output (single tensor)
          - branch="txt": hook block.ff_context output (single tensor)
        """
        def hook(module, input, output):
            step = state["step"]
            if args.block_steering != 'all' and layer_idx not in args.block_steering:
                return output
            if args.t_steering != 'all' and step not in args.t_steering:
                return output

            print(f"step: {step}, layer_idx: {layer_idx}, branch: {branch}")
            _, sv_img, sv_txt = _get_layer_vectors(step, layer_idx)
            sv = sv_img if branch == "img" else sv_txt
            if sv is None:
                return output

            if not torch.is_tensor(output):
                return output

            to_modify = output.clone()
            score_val = _get_score_val(step, layer_idx, to_modify, models, scores_all, eigen_info, cfg, args)
            strength = args.strength_img if branch == "img" else args.strength
            return _apply_to_branch(to_modify, sv, args, score_val, strength)

        return hook

    # Hook registration — identical to your original
    if args.injection_point in ("attn", "block", "ff") and hasattr(pipe.transformer, "transformer_blocks"):
        blocks = pipe.transformer.transformer_blocks
        for layer_id, block in enumerate(blocks):
            if args.injection_point == "attn":
                hook_handles.append(block.attn.register_forward_hook(steering_hook(layer_id)))
            elif args.injection_point == "block":
                hook_handles.append(block.register_forward_hook(steering_hook(layer_id)))
            elif args.injection_point == "ff":
                hook_handles.append(block.ff.register_forward_hook(steering_hook_ff(layer_id, "img")))
                hook_handles.append(block.ff_context.register_forward_hook(steering_hook_ff(layer_id, "txt")))
    else:
        # Fallback (non-FLUX / older structure): preserve original attn discovery behavior
        layer_id = 0
        for name, module in pipe.transformer.named_modules():
            if name.endswith("attn"):
                hook_handles.append(module.register_forward_hook(steering_hook(layer_id)))
                layer_id += 1

    return state, lambda: [h.remove() for h in hook_handles]


# ==============================================================================
# Main — same as your original, with added --strength_img
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--prompts_path', type=str, default='data/captions.txt')
    parser.add_argument('--vector_type', type=str, default='diff')
    parser.add_argument('--num_prompts', type=int, default=500)
    parser.add_argument('--task', type=str, default='add concept', choices=['add concept', 'remove', 'nudity'])
    parser.add_argument('--remove_prompt', type=str, default='cyberpunk style')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)

    # Steering Config
    parser.add_argument('--strength', type=float, default=20.0, help="Text branch strength (inside transformer)")
    parser.add_argument('--strength_img', type=float, default=20.0, help="Image branch strength (NEW)")
    parser.add_argument('--steering_type', type=str, default='mean')
    parser.add_argument(
        '--injection_point',
        type=str,
        default='attn',
        choices=['attn', 'block', 'ff'],
        help="Where to inject steering inside the transformer (must match how activations were extracted).",
    )
    parser.add_argument('--use_ssim_mask', action='store_true')
    parser.add_argument('--top_k_percent', type=float, default=0.01)
    parser.add_argument('--use_cls', action='store_true')
    parser.add_argument('--min_signal_threshold', type=float, default=0.5)

    # Steering Hyperparams
    parser.add_argument('--block_steering', type=str, default='all')
    parser.add_argument('--t_steering', type=str, default='all')
    parser.add_argument('--quantile_level', type=float, default=0.5)
    parser.add_argument('--quantile_type', type=str, default='0.85')

    # Generation
    parser.add_argument('--steer_txt', action='store_true', help="Also steer T5 encoder output (separate file)")
    parser.add_argument('--strength_txt', type=float, default=2.0, help="T5 encoder steering strength")
    parser.add_argument('--inference_steps', type=int, default=4)
    parser.add_argument('--guidance_scale', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='results_steered')
    parser.add_argument('--structure', type=float, default=0.5)
    parser.add_argument('--block_structure', type=int, default=15)
    parser.add_argument('--t_structure', type=int, default=0)
    parser.add_argument('--cls_min', type=float, default=21.0)
    parser.add_argument('--cls_type', type=str, default='tanh')

    args = parser.parse_args()

    if args.block_steering != 'all':
        args.block_steering = [int(x) for x in args.block_steering.split(',')]
    if args.t_steering != 'all':
        args.t_steering = [int(x) for x in args.t_steering.split(',')]

    # Pipeline
    is_flux = 'flux' in args.model_name.lower()
    pipe = (FluxPipeline if is_flux else StableDiffusion3Pipeline).from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16 if is_flux else torch.float16,
        device_map="balanced", use_safetensors=True,
    )

    # --- Load main vector (img+txt transformer branches) ---
    vector_suffix_candidates = [f"_{args.vector_type}.pt", f"{args.vector_type}.pt"]
    vector_candidates = [
        os.path.join(args.data_dir, name)
        for name in os.listdir(args.data_dir)
        if name.endswith(".pt")
        and "text_" not in name
        and any(name.endswith(suf) for suf in vector_suffix_candidates)
    ]
    if not vector_candidates:
        raise FileNotFoundError(f"No vector file in '{args.data_dir}' ending with '{args.vector_type}.pt'")
    if len(vector_candidates) > 1:
        raise RuntimeError(f"Multiple candidates: {vector_candidates}")

    vector = torch.load(vector_candidates[0])
    print(f"Loaded: {vector_candidates[0]}")

    # --- Text encoder steering (T5 sequence + CLIP pooled) ---
    vector_txt = None
    if args.steer_txt:
        vector_name = os.path.basename(vector_candidates[0])
        vector_no_ext = vector_name[:-len(".pt")]
        if vector_no_ext.endswith(f"_{args.vector_type}"):
            prefix = vector_no_ext[:-len(f"_{args.vector_type}")]
        elif vector_no_ext.endswith(args.vector_type):
            prefix = vector_no_ext[:-len(args.vector_type)]
        else:
            raise RuntimeError(f"Cannot derive prefix from '{vector_name}'")
        prefix = prefix.rstrip("_")

        expected_txt_names = [
            f"{prefix}_text_{args.vector_type}.pt",
            f"{prefix}text_{args.vector_type}.pt",
        ]
        txt_candidates = [
            os.path.join(args.data_dir, n) for n in expected_txt_names
            if os.path.exists(os.path.join(args.data_dir, n))
        ]
        if txt_candidates:
            vector_txt = torch.load(txt_candidates[0])
            print(f"Text encoder vector: {txt_candidates[0]}")
        else:
            print(f"WARNING: --steer_txt set but no text file found. Tried: {expected_txt_names}")

    txt_steering = {
        'vector': vector_txt,
        'strength': args.strength_txt,
        'task': args.task,
    }

    # --- Prompts ---
    try:
        with open(args.prompts_path, "r") as f:
            coco_prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        coco_prompts = ["A high quality photo"]

    if args.task == 'nudity':
        dataset = load_dataset("AIML-TUDA/i2p", split="train")
        coco_prompts = [sample["prompt"] for sample in dataset]
        coco_seeds = [int(sample["sd_seed"]) for sample in dataset]
    else:
        coco_seeds = [int(args.seed) for _ in range(len(coco_prompts))]

    # --- Generate ---
    for idx, prompt in enumerate(coco_prompts):
        if args.task == 'remove':
            prompt = prompt + " " + args.remove_prompt

        sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        suffix = f"s_{args.strength}_simg_{args.strength_img}_mask_{args.use_ssim_mask}_v_{args.vector_type}"
        print(prompt)
        out_path = os.path.join(args.results_dir, 'steered', f"{idx:02d}_{sanitized}_{suffix}.png")
        if os.path.exists(out_path):
            #assert False
            continue

        seed = int(coco_seeds[idx])
        print(f"{idx + 1}/{len(coco_prompts)}: {prompt[:60]}... seed={seed}")

        hook_state, remove_hooks = apply_attention_steering(pipe, args, vector)
        generator = torch.Generator().manual_seed(seed)

        def _on_step_end(_pipe, i, _t, callback_kwargs):
            # Keep hook step in sync with diffusion loop:
            # current forward pass uses step i, next pass should see i+1.
            hook_state.update({"step": int(i) + 1})
            return callback_kwargs


        images = pipe(
            prompt, num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale, width=args.width, height=args.height,
            generator=generator, structure_strength=args.structure,
            callback_on_step_end=_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
            #callback=lambda step, **k: hook_state.update({"step": step}),
            #callback_steps=1, 
            txt_steering=txt_steering,
        ).images
        #assert False
        os.makedirs(os.path.join(args.results_dir, 'steered'), exist_ok=True)
        try:
            images[0].save(out_path)
        except Exception as exc:
            print(f"Save failed idx {idx}: {exc}")
        
        if len(images) > 1:
            os.makedirs(os.path.join(args.results_dir, 'origin'), exist_ok=True)
            images[1].save(os.path.join(args.results_dir, 'origin', f"{idx:02d}_orig.png"))

        remove_hooks()
        # if idx >= 5:
        #     assert False