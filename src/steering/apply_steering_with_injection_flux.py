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
            return mask, weight.clip(0,2)
        
        dtype = activations.dtype
        act_f32 = activations.float()
        orig_norm = torch.norm(act_f32, dim=-1, keepdim=True) + 1e-6
        act_unit = act_f32 / orig_norm
        
        v_unit = steering_vec.float() / (torch.norm(steering_vec.float(), dim=-1, keepdim=True) + 1e-6)
        v_steer = v_unit

        if args.use_ssim_mask:
            if args.vector_type == 'diff' or args.steering_type == 'mean':
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
            if args.vector_type == 'diff' or args.steering_type == 'mean':
                adjustment = args.strength * weight * v_steer.to(activations.dtype) * mask * score_val
            else:
                score_val = score_val.unsqueeze(1)
                adjustment = args.strength * weight * v_steer.to(activations.dtype) * mask * score_val
                adjustment = adjustment.mean(0)
            steered = activations - (adjustment)
        else:
            # Masked addition targets concept-relevant tokens
            if args.vector_type == 'diff' or args.steering_type == 'mean':
                adjustment = args.strength * v_steer.to(activations.dtype) * score_val  
            else:
                score_val = score_val.unsqueeze(1)
                adjustment = args.strength * v_steer.to(activations.dtype).unsqueeze(1) * mask * score_val.unsqueeze(1)
                adjustment = adjustment.mean(0)
            steered = activations + (adjustment)

        # 4. Energy Restoration
        steered_unit = steered.float() / (torch.norm(steered.float(), dim=-1, keepdim=True) + 1e-6)
        return (steered_unit * orig_norm).to(dtype)


MODEL_CONFIGS = {
    "flux": {"last_layer": 66, "out_idx": 1, "inner_idx": 0, "dtype": torch.bfloat16},
    "sd3": {"last_layer": 23, "out_idx": 1, "inner_idx": 1, "dtype": torch.float16}
}

def apply_attention_steering(pipe, args, vector, idx=None):
    m_key = "flux" if "flux" in args.model_name.lower() else "sd3"
    cfg = MODEL_CONFIGS[m_key]
    state = {"step": 0}
    hook_handles = []

    # Map your paths from argparse
    eigen_path = os.path.join(args.data_dir, f"cos_sep.pt")
    svm_path = os.path.join(args.data_dir, f"base_0.85_20_svm_models.pt")
    scr_path = os.path.join(args.data_dir, f"base_0.85_20_scores.pt")
    tok_path = os.path.join(args.data_dir, f"base_0.85_20_tokens.pt")

    models, tokens_best, scores_all, quantiles = load_steering_data(
        svm_path, scr_path, tok_path, args.quantile_level
    )
   
    try:
        eigen_info = torch.load(eigen_path, weights_only=False)
    except:
        eigen_info = None

    def steering_hook(layer_idx: int):
        def hook(module, input, output):
           
            step = state["step"]
            if layer_idx == cfg["last_layer"]: state["step"] += 1

            if step not in vector or f"layer_{layer_idx}" not in vector[step]: return output
            if args.block_steering != 'all' and layer_idx not in args.block_steering: return output
            if args.t_steering != 'all' and step not in args.t_steering: return output
            
            if len(output) != 2: return output
            
            if eigen_info is not None:
                
                eigen = eigen_info[step][f'layer_{layer_idx}']
            else:
                eigen = 1

            act_tuple = list(output)
            hidden_states = act_tuple[cfg["out_idx"]]
            to_modify = hidden_states[cfg["inner_idx"]].clone()

            # Dynamic Classifier Scoring & Fallback
            score_val = torch.ones((1, 1), device=to_modify.device, dtype=to_modify.dtype)
            
            current_signal = 1.0 # Default
            if scores_all is not None:
                layer_scores = scores_all[:, step, layer_idx]
                current_signal = layer_scores.float()

            # 2. Logic: If signal is low, skip cls and use score=1
            if args.use_cls and models:
                ensemble = models.get(step, {}).get(f"layer_{layer_idx}")
                if ensemble:
                    mean_act = (to_modify).mean(0, keepdim=True)  
                    mean_act = mean_act / (mean_act.norm(dim=-1, keepdim=True) + 1e-6)
                    votes = [
                        calculate_cls_score(mean_act.cpu().float(), args.cls_min, m, args.cls_type, task=args.task, use_distance=False)[0] 
                        if current_signal[i] > args.min_signal_threshold else 1.0 
                        for i, m in enumerate(ensemble)
                    ]
                    score_val = torch.tensor(votes).to(to_modify.device, to_modify.dtype)
                    if args.vector_type == 'diff' or args.steering_type == 'mean':
                        score_val = torch.mean(score_val, dim=0, keepdim=True)
            else:
                score_val = torch.ones((1, 1), device=to_modify.device, dtype=to_modify.dtype)

            print(score_val)
            if args.steering_type == 'mean':
                
                steering_vec = (vector[step][f"layer_{layer_idx}"]).mean(0, keepdim=True).to(to_modify.device)
            else:
                steering_vec = (vector[step][f"layer_{layer_idx}"]).to(to_modify.device, to_modify.dtype)
            final_act = SteeringEngine.apply_steering(to_modify, steering_vec, args, score_val)

            act_tuple[cfg["out_idx"]][cfg["inner_idx"]] = final_act
            return tuple(act_tuple)
        return hook

    # Hook Registration
    
    layer_id = 0
    for name, module in pipe.transformer.named_modules():
        if name.endswith("attn"):
            hook_handles.append(module.register_forward_hook(steering_hook(layer_id)))
            layer_id += 1
            
    return state, lambda: [h.remove() for h in hook_handles]

# ==============================================================================
# 4. Main Loop
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
    parser.add_argument('--strength', type=float, default=20.0)
    parser.add_argument('--steering_type', type=str, default='mean')
    parser.add_argument('--use_ssim_mask', action='store_true', help="Use token similarity masking for both tasks")
    parser.add_argument('--top_k_percent', type=float, default=0.01)
    parser.add_argument('--use_cls', action='store_true')
    parser.add_argument('--min_signal_threshold', type=float, default=0.5)
    
    # Steering Hyperparams
    parser.add_argument('--block_steering', type=str, default='all')
    parser.add_argument('--t_steering', type=str, default='all')
    parser.add_argument('--quantile_level', type=float, default=0.5)
    parser.add_argument('--quantile_type', type=str, default='0.85')

    # Generation Settings
    parser.add_argument('--steer_txt', action='store_true')
    parser.add_argument('--strength_txt', type=float, default=2.0)
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

    # Parse Filters
    if args.block_steering != 'all': args.block_steering = [int(x) for x in args.block_steering.split(',')]
    if args.t_steering != 'all': args.t_steering = [int(x) for x in args.t_steering.split(',')]

    # Pipeline
    is_flux = 'flux' in args.model_name.lower()
    pipe = (FluxPipeline if is_flux else StableDiffusion3Pipeline).from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16 if is_flux else torch.float16, device_map="balanced",  use_safetensors=True
    )

    vector_suffix_candidates = [f"_{args.vector_type}.pt", f"{args.vector_type}.pt"]
    vector_candidates = [
        os.path.join(args.data_dir, name)
        for name in os.listdir(args.data_dir)
        if name.endswith(".pt")
        and "text_" not in name
        and any(name.endswith(suf) for suf in vector_suffix_candidates)
    ]
    if len(vector_candidates) == 0:
        raise FileNotFoundError(
            f"No vector file found in '{args.data_dir}' ending with '{args.vector_type}.pt'."
        )
    if len(vector_candidates) > 1:
        raise RuntimeError(f"Multiple vector candidates found: {vector_candidates}")
    vector = torch.load(vector_candidates[0])

    if args.steer_txt:
        vector_name = os.path.basename(vector_candidates[0])  
        vector_no_ext = vector_name[: -len(".pt")]

        if vector_no_ext.endswith(f"_{args.vector_type}"):
            prefix = vector_no_ext[: -len(f"_{args.vector_type}")]
        elif vector_no_ext.endswith(args.vector_type):
            prefix = vector_no_ext[: -len(args.vector_type)]
        else:
            raise RuntimeError(
                f"Cannot derive prefix from vector filename '{vector_name}' for vector_type='{args.vector_type}'."
            )

        prefix = prefix.rstrip("_")

        expected_txt_names = [
            f"{prefix}_text_diff.pt",
            f"{prefix}text_diff.pt",
        ]
        vector_txt_candidates = [
            os.path.join(args.data_dir, n)
            for n in expected_txt_names
            if os.path.exists(os.path.join(args.data_dir, n))
        ]
        if len(vector_txt_candidates) == 0:
            raise FileNotFoundError(
                f"No text vector derived from vector '{vector_name}'. Tried: {expected_txt_names} in '{args.data_dir}'."
            )
        if len(vector_txt_candidates) > 1:
            raise RuntimeError(f"Multiple derived text vector candidates found: {vector_txt_candidates}")
        vector_txt = torch.load(vector_txt_candidates[0])
    
    txt_steering = (
        {'vector': vector_txt, 'strength': args.strength_txt, 'task': args.task}
        if args.steer_txt
        else {'vector': None, 'strength': args.strength_txt, 'task': args.task}
    )

    try:
        with open(args.prompts_path, "r") as f:
            coco_prompts = [line.strip() for line in f if line.strip()][:]
    except FileNotFoundError:
        coco_prompts = ["A high quality photo"]
    if args.task == 'nudity':
        dataset = load_dataset("AIML-TUDA/i2p", split="train")
        coco_prompts = [sample["prompt"] for sample in dataset]
        coco_seeds = [int(sample["sd_seed"]) for sample in dataset]
    else:
        # For non-nudity tasks, keep prompts from file and use a constant seed.
        coco_seeds = [int(args.seed) for _ in range(len(coco_prompts))]
    # Main Loop
    for idx, prompt in enumerate(coco_prompts):
        if args.task == 'remove':
            prompt = prompt + " " + args.remove_prompt

        sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        suffix = f"s_{args.strength}_mask_{args.use_ssim_mask}_v_{args.vector_type}"
        
        if os.path.exists(os.path.join(f'{args.results_dir}', 'steered', f"{idx:02d}_{sanitized}_{suffix}.png")):
            continue
        else:
            print(os.path.join(f'{args.results_dir}', 'steered', f"{idx:02d}_{sanitized}_{suffix}.png"))
        seed = int(coco_seeds[idx])
        print(f"Processing {idx+1}/{len(coco_prompts)}: {prompt}", seed,  args.inference_steps, args.guidance_scale)
        
        hook_state, remove_hooks = apply_attention_steering(pipe, args, vector)
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator().manual_seed(seed)
        
        images = pipe(
            prompt, num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale, width=args.width, height=args.height,
            generator=generator, structure_strength=args.structure,
            callback=lambda step, **k: hook_state.update({"step": step}), callback_steps=1,
            txt_steering=txt_steering
        ).images
        sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        suffix = f"s_{args.strength}_mask_{args.use_ssim_mask}_v_{args.vector_type}"
        os.makedirs(os.path.join(f'{args.results_dir}', 'steered'), exist_ok=True)
        try:
            images[0].save(os.path.join(f'{args.results_dir}', 'steered', f"{idx:02d}_{sanitized}_{suffix}.png"))
        except Exception as exc:
            print(f"Failed to save steered image for index {idx}: {exc}")
        if len(images) > 1:
            os.makedirs(os.path.join(args.results_dir, 'origin'), exist_ok=True)
            images[1].save(os.path.join(args.results_dir, 'origin', f"{idx:02d}_orig.png"))

        remove_hooks()