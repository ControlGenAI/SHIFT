"""
Apply dual-branch steering to SD3.5 MM-DiT transformer blocks.

Expected vector format:
  vector[step]["layer_N"] = {"img": tensor, "txt": tensor}

Supported injection points:
  - block: JointTransformerBlock output (txt, img)
  - attn:  block.attn output (img, txt)
  - ff:    block.ff for img and block.ff_context for txt when present
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sklearn.svm._classes
import torch
import torch.nn.functional as F
from datasets import load_dataset

STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))

from src.models.sd_3 import StableDiffusion3Pipeline
from src.utils.utils import calculate_cls_score, pooled_cosine_multiplier_from_vector


def vector_is_dense_per_token(vector_type: str) -> bool:
    return vector_type == "diff" or vector_type.startswith("ot_") or "ot" in vector_type


def split_vector(vector: dict) -> Tuple[dict, dict]:
    img_vec, txt_vec = {}, {}
    is_dual = False
    for step in vector:
        if not isinstance(step, int):
            continue
        for layer_key, val in vector[step].items():
            if isinstance(val, dict) and ("img" in val or "txt" in val):
                is_dual = True
            break
        break

    if not is_dual:
        return {}, vector

    for step in vector:
        if not isinstance(step, int):
            continue
        img_vec[step], txt_vec[step] = {}, {}
        for layer_key, val in vector[step].items():
            if not isinstance(val, dict):
                txt_vec[step][layer_key] = val
                continue
            if "img" in val and val["img"] is not None:
                img_vec[step][layer_key] = squeeze_vector(val["img"])
            if "txt" in val and val["txt"] is not None:
                # print(val["txt"].shape)
                # assert False, "val['txt'].shape"
                txt_vec[step][layer_key] = squeeze_vector(val["txt"])
    
    return img_vec, txt_vec


def squeeze_vector(tensor: torch.Tensor) -> torch.Tensor:
    while torch.is_tensor(tensor) and tensor.dim() > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _find_one_file(data_dir: str, suffix: str) -> Optional[str]:
    legacy = os.path.join(data_dir, suffix)
    if os.path.exists(legacy):
        return legacy
    matches = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(suffix)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files matching *{suffix} in {data_dir}: {matches}")
    return None


def load_optional_steering_data(data_dir: str, quantile_level: float):
    svm_path = _find_one_file(data_dir, "_svm_models.pt") or _find_one_file(data_dir, "svm_models.pt")
    scores_path = _find_one_file(data_dir, "_scores.pt")
    eigen_path = os.path.join(data_dir, "cos_sep.pt")

    models = None
    if svm_path and os.path.exists(svm_path):
        with torch.serialization.safe_globals([sklearn.svm._classes.SVC]):
            models = torch.load(svm_path, weights_only=False)

    scores = torch.load(scores_path, weights_only=False) if scores_path and os.path.exists(scores_path) else None
    quantiles = None
    if scores is not None and torch.is_tensor(scores):
        quantiles = np.quantile(scores.numpy(), q=quantile_level)

    try:
        eigen_info = torch.load(eigen_path, weights_only=False)
    except Exception:
        eigen_info = None

    return models, scores, quantiles, eigen_info


class SteeringEngine:
    @classmethod
    def apply_steering(
        cls,
        activations: torch.Tensor,
        steering_vec: torch.Tensor,
        args,
        score_val: torch.Tensor,
        condition_vec: Optional[torch.Tensor] = None,
        cast_threshold: Optional[float] = None,
        cast_comparator: Optional[str] = None,
        pooled_scale: float = 1.0,
    ) -> torch.Tensor:
        def calculate_sim(act_unit, v_unit):
            sim = F.cosine_similarity(act_unit, v_unit, dim=-1)
            k_val = max(1, int(sim.numel() * args.top_k_percent))
            threshold = torch.topk(sim.flatten(), k_val).values[-1]
            mask = (sim >= threshold).float().unsqueeze(-1)
            return mask, sim.unsqueeze(-1).clip(0, 2)

        def calculate_cast_gate(act_f32, condition_vec, threshold, comparator):
            token_dim = 1 if act_f32.dim() >= 3 else 0
            h = act_f32.mean(dim=token_dim)
            if h.dim() == 1:
                h = h.unsqueeze(0)
            c = condition_vec.float().to(h.device).flatten()
            c_norm_sq = torch.dot(c, c).clamp(min=1e-6)
            proj = (h @ c / c_norm_sq).unsqueeze(-1) * c.unsqueeze(0)
            proj_tanh = torch.tanh(proj)
            h_norm = h.norm(dim=-1).clamp(min=1e-6)
            proj_norm = proj_tanh.norm(dim=-1).clamp(min=1e-6)
            cast_sim = (h * proj_tanh).sum(dim=-1) / (h_norm * proj_norm)
            if comparator == "smaller":
                gate = cast_sim > threshold
            elif comparator == "larger":
                gate = cast_sim < threshold
            else:
                raise ValueError(f"Unknown cast_comparator: {comparator}")
            return gate.to(act_f32.dtype).view(-1, *([1] * (act_f32.dim() - 1)))

        dtype = activations.dtype
        act_f32 = activations.float()
        orig_norm = act_f32.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        act_unit = act_f32 / orig_norm
        v_unit = steering_vec.float() / steering_vec.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)

        if args.use_ssim_mask:
            if vector_is_dense_per_token(args.vector_type) or args.steering_type == "mean":
                mask, weight = calculate_sim(act_unit, v_unit)
            else:
                masks, weights = [], []
                for v in v_unit:
                    mask_i, weight_i = calculate_sim(act_unit, v)
                    masks.append(mask_i)
                    weights.append(weight_i)
                mask, weight = torch.stack(masks), torch.stack(weights)
        else:
            mask, weight = 1.0, 1.0

        if args.steering_mode == "cast":
            if condition_vec is None:
                raise ValueError("--steering_mode cast requires a matching condition vector")
            cast_gate = calculate_cast_gate(
                act_f32,
                condition_vec,
                args.cast_threshold if cast_threshold is None else cast_threshold,
                args.cast_comparator if cast_comparator is None else cast_comparator,
            )
        else:
            cast_gate = 1.0

        scale_factor = pooled_scale
        if not torch.is_tensor(scale_factor):
            scale_factor = torch.tensor(scale_factor, device=activations.device, dtype=dtype)
        else:
            scale_factor = scale_factor.to(device=activations.device, dtype=dtype)
        if getattr(args, "use_pooled_cosine_score", False):
            scale_factor = scale_factor.clip(0, 0.3)

        if args.task in ("remove", "nudity"):
            if vector_is_dense_per_token(args.vector_type) or args.steering_type == "mean":
                print(weight, score_val, cast_gate, scale_factor)
                print(v_unit.shape)
                # assert False, "v_unit.shape"
                adjustment = args.strength * weight * v_unit.to(dtype) * mask * score_val * cast_gate * scale_factor
            else:
                adjustment = args.strength * weight * v_unit.to(dtype) * mask * score_val.unsqueeze(1) * cast_gate * scale_factor
                adjustment = adjustment.mean(0)
            steered = activations - adjustment
        else:
            if vector_is_dense_per_token(args.vector_type) or args.steering_type == "mean":
                adjustment = args.strength * v_unit.to(dtype) * score_val * cast_gate * scale_factor
            else:
                adjustment = args.strength * v_unit.to(dtype).unsqueeze(1) * mask * score_val.unsqueeze(1) * cast_gate * scale_factor
                adjustment = adjustment.mean(0)
            steered = activations + adjustment

        steered_unit = steered.float() / steered.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return (steered_unit * orig_norm).to(dtype)


def apply_attention_steering_sd35(pipe, args, vector, pooled_scale: float = 1.0):
    state = {"step": 0, "pooled_scale": pooled_scale}
    hook_handles = []
    img_vector, txt_vector = split_vector(vector)
    condition_img_vector, condition_txt_vector = {}, {}
    cast_thresholds: Dict = {}

    if args.steering_mode == "cast":
        if not args.condition_vector_path:
            raise ValueError("--condition_vector_path is required when --steering_mode cast")
        condition_vector = torch.load(args.condition_vector_path, map_location="cpu")
        condition_img_vector, condition_txt_vector = split_vector(condition_vector)
        threshold_path = args.cast_threshold_path
        if threshold_path is None:
            candidate = args.condition_vector_path.replace(".pt", "_thresholds.pt")
            threshold_path = candidate if os.path.exists(candidate) else None
        if threshold_path:
            cast_thresholds = torch.load(threshold_path, map_location="cpu")

    models, scores_all, _, eigen_info = load_optional_steering_data(args.data_dir, args.quantile_level)
    print(f"  Steering branches: img={'YES' if img_vector else 'NO'}, txt={'YES' if txt_vector else 'NO'}")

    def _target_slice(hidden: torch.Tensor):
        if args.steer_batch_slice == "all":
            return slice(None)
        batch = max(1, getattr(args, "_current_batch_size", 1))
        if hidden.shape[0] >= 2 * batch:
            return slice(batch, 2 * batch)
        return slice(0, batch)

    def _get_score_val(step, layer_idx, branch, to_modify):
        score_val = torch.ones((1, 1), device=to_modify.device, dtype=to_modify.dtype)
        current_signal = None
        if torch.is_tensor(scores_all) and scores_all.dim() >= 3:
            current_signal = scores_all[:, step, layer_idx].float()
        elif isinstance(scores_all, dict):
            current_signal = scores_all.get(step, {}).get(f"layer_{layer_idx}", {}).get(branch)
            if current_signal is not None:
                current_signal = torch.tensor(current_signal).float()

        if args.use_cls and models:
            ensemble = models.get(step, {}).get(f"layer_{layer_idx}")
            if isinstance(ensemble, dict):
                ensemble = ensemble.get(branch)
            if ensemble:
                mean_act = to_modify.mean(0).mean(0, keepdim=True)
                mean_act = mean_act / mean_act.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                votes = []
                for i, model in enumerate(ensemble):
                    signal_ok = True if current_signal is None else current_signal[i] > args.min_signal_threshold
                    vote = calculate_cls_score(
                        mean_act.cpu().float(), args.cls_min, model, args.cls_type, task=args.task, use_distance=False
                    )[0] if signal_ok else 1.0
                    votes.append(vote)
                score_val = torch.tensor(votes, device=to_modify.device, dtype=to_modify.dtype)
                if vector_is_dense_per_token(args.vector_type) or args.steering_type == "mean":
                    score_val = score_val.mean(dim=0, keepdim=True)
        return score_val

    def _get_vectors(step: int, layer_idx: int):
        layer_key = f"layer_{layer_idx}"
        sv_img = img_vector.get(step, {}).get(layer_key)
        sv_txt = txt_vector.get(step, {}).get(layer_key)
        return sv_img, sv_txt

    def _get_condition(step: int, layer_idx: int):
        if args.steering_mode != "cast":
            return None, None, None, None
        condition_layer_idx = args.cast_layer if args.cast_layer >= 0 else layer_idx
        layer_key = f"layer_{condition_layer_idx}"
        c_img = condition_img_vector.get(step, {}).get(layer_key)
        c_txt = condition_txt_vector.get(step, {}).get(layer_key)
        threshold_layer = cast_thresholds.get(step, {}).get(layer_key, {})
        return c_img, c_txt, threshold_layer.get("img"), threshold_layer.get("txt")

    def _apply_branch(hidden, sv_raw, branch, step, layer_idx, strength, condition_raw=None, cast_config=None):
        if sv_raw is None or not torch.is_tensor(hidden):
            return hidden
        result = hidden.clone()
        target = _target_slice(result)
        to_modify = result[target].clone()
        print(to_modify.shape, sv_raw.shape)
        steering_vec = sv_raw.mean(0, keepdim=True).to(to_modify.device) if args.steering_type == "mean" else sv_raw.to(to_modify.device, to_modify.dtype)
        condition_vec = condition_raw.to(to_modify.device).float() if condition_raw is not None else None
        score_val = _get_score_val(step, layer_idx, branch, to_modify)
        old_strength = args.strength
        args.strength = strength
        result[target] = SteeringEngine.apply_steering(
            to_modify,
            steering_vec,
            args,
            score_val,
            condition_vec=condition_vec,
            cast_threshold=cast_config.get("threshold") if cast_config else None,
            cast_comparator=cast_config.get("comparator") if cast_config else None,
            pooled_scale=state["pooled_scale"],
        )
        args.strength = old_strength
        return result

    def steering_hook(layer_idx: int):
        def hook(module, input, output):
            step = state["step"]
            if args.block_steering != "all" and layer_idx not in args.block_steering:
                return output
            if args.t_steering != "all" and step not in args.t_steering:
                return output

            sv_img, sv_txt = _get_vectors(step, layer_idx)
            c_img, c_txt, t_img, t_txt = _get_condition(step, layer_idx)
            if sv_img is None and sv_txt is None:
                return output

            if args.injection_point == "block":
                if not (isinstance(output, tuple) and len(output) >= 2):
                    return output
                txt_hidden, img_hidden = output[0], output[1]
                txt_new = _apply_branch(txt_hidden, sv_txt, "txt", step, layer_idx, args.strength, c_txt, t_txt)
                img_new = _apply_branch(img_hidden, sv_img, "img", step, layer_idx, args.strength_img, c_img, t_img)
                return (txt_new, img_new)

            if args.injection_point == "attn":
                if not (isinstance(output, tuple) and len(output) >= 2):
                    return output
                img_hidden, txt_hidden = output[0], output[1]
                img_new = _apply_branch(img_hidden, sv_img, "img", step, layer_idx, args.strength_img, c_img, t_img)
                txt_new = _apply_branch(txt_hidden, sv_txt, "txt", step, layer_idx, args.strength, c_txt, t_txt)
                return (img_new, txt_new)

            return output
        return hook

    def steering_hook_ff(layer_idx: int, branch: str):
        def hook(module, input, output):
            step = state["step"]
            if args.block_steering != "all" and layer_idx not in args.block_steering:
                return output
            if args.t_steering != "all" and step not in args.t_steering:
                return output
            sv_img, sv_txt = _get_vectors(step, layer_idx)
            c_img, c_txt, t_img, t_txt = _get_condition(step, layer_idx)
            if branch == "img":
                return _apply_branch(output, sv_img, "img", step, layer_idx, args.strength_img, c_img, t_img)
            return _apply_branch(output, sv_txt, "txt", step, layer_idx, args.strength, c_txt, t_txt)
        return hook

    for layer_id, block in enumerate(pipe.transformer.transformer_blocks):
        if args.injection_point == "attn":
            hook_handles.append(block.attn.register_forward_hook(steering_hook(layer_id)))
        elif args.injection_point == "block":
            hook_handles.append(block.register_forward_hook(steering_hook(layer_id)))
        elif args.injection_point == "ff":
            hook_handles.append(block.ff.register_forward_hook(steering_hook_ff(layer_id, "img")))
            if hasattr(block, "ff_context") and block.ff_context is not None:
                hook_handles.append(block.ff_context.register_forward_hook(steering_hook_ff(layer_id, "txt")))
        else:
            raise ValueError(f"Unknown injection_point: {args.injection_point}")

    return state, lambda: [handle.remove() for handle in hook_handles]


def find_vector_file(data_dir: str, vector_type: str) -> str:
    suffixes = [f"_{vector_type}.pt", f"{vector_type}.pt"]
    candidates = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".pt") and "text_" not in name and any(name.endswith(suffix) for suffix in suffixes)
    ]
    if not candidates:
        raise FileNotFoundError(f"No vector file in '{data_dir}' ending with '{vector_type}.pt'")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple vector candidates: {candidates}")
    return candidates[0]


def find_text_vector_file(data_dir: str, vector_path: str, vector_type: str, explicit_path: Optional[str] = None) -> Optional[str]:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"--txt_vector_path does not exist: {explicit_path}")
        return explicit_path

    vector_name = os.path.basename(vector_path)
    vector_no_ext = vector_name[:-len(".pt")] if vector_name.endswith(".pt") else vector_name
    if vector_no_ext.endswith(f"_{vector_type}"):
        prefix = vector_no_ext[:-len(f"_{vector_type}")].rstrip("_")
    elif vector_no_ext.endswith(vector_type):
        prefix = vector_no_ext[:-len(vector_type)].rstrip("_")
    else:
        prefix = vector_no_ext.rstrip("_")

    expected_names = [
        f"{prefix}_text_{vector_type}.pt",
        f"{prefix}text_{vector_type}.pt",
        f"{prefix}_text_diff.pt",
        f"{prefix}text_diff.pt",
    ]
    expected = [
        os.path.join(data_dir, name)
        for name in expected_names
        if os.path.exists(os.path.join(data_dir, name))
    ]
    if expected:
        return expected[0]

    matches = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".pt") and "text" in name and name.endswith(f"_{vector_type}.pt")
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple text vector candidates: {matches}")
    return None


def load_prompts_and_seeds(args):
    if args.task == "nudity":
        dataset = load_dataset("AIML-TUDA/i2p", split="train")
        return [sample["prompt"] for sample in dataset], [int(sample["sd_seed"]) for sample in dataset]

    try:
        with open(args.prompts_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        prompts = ["A high quality photo"]

    if args.seeds_path and os.path.exists(args.seeds_path):
        with open(args.seeds_path, "r") as f:
            seeds = [int(line.strip()) for line in f if line.strip()]
    else:
        assert False
        seeds = [int(args.seed) for _ in prompts]
    return prompts, seeds


def _pooled_cos_scalar(cos_coeff) -> float:
    if torch.is_tensor(cos_coeff):
        return float(cos_coeff.reshape(-1)[0].item())
    return float(cos_coeff)


def compute_pooled_cos_for_prompt(pipe, prompt: str, vector_txt, args) -> float:
    _, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        device=pipe._execution_device,
        do_classifier_free_guidance=False,
        num_images_per_prompt=1,
    )
    cos_coeff = pooled_cosine_multiplier_from_vector(
        pooled_prompt_embeds,
        vector_txt,
        task=args.task,
        mode=args.pooled_cosine_score_mode,
    )
    return _pooled_cos_scalar(cos_coeff)


def collect_pooled_cosine_results(pipe, prompts: List[str], vector_txt, args) -> Tuple[List[Dict], List[float]]:
    cos_values: List[float] = []
    prompt_results: List[Dict] = []
    for idx, prompt in enumerate(prompts):
        cos_val = compute_pooled_cos_for_prompt(pipe, prompt, vector_txt, args)
        cos_values.append(cos_val)
        prompt_results.append({"idx": idx, "prompt": prompt, "pooled_cos": cos_val})
    return prompt_results, cos_values


def compute_pooled_cos_stats(cos_values: List[float]) -> Dict[str, Optional[float]]:
    if not cos_values:
        return {"min": None, "max": None, "mean": None, "median": None, "count": 0}
    cos_arr = np.asarray(cos_values, dtype=np.float64)
    return {
        "min": float(cos_arr.min()),
        "max": float(cos_arr.max()),
        "mean": float(cos_arr.mean()),
        "median": float(np.median(cos_arr)),
        "count": int(cos_arr.size),
    }


def build_pooled_cos_groups(prompt_results: List[Dict], group_size: int = 20) -> List[Dict]:
    groups = []
    for start in range(0, len(prompt_results), group_size):
        chunk = prompt_results[start:start + group_size]
        if not chunk:
            continue
        cos_vals = [entry["pooled_cos"] for entry in chunk]
        groups.append(
            {
                "range": f"{chunk[0]['idx']}-{chunk[-1]['idx']}",
                "start_idx": chunk[0]["idx"],
                "end_idx": chunk[-1]["idx"],
                "num_prompts": len(chunk),
                "stats": compute_pooled_cos_stats(cos_vals),
                "coefficients": cos_vals,
                "prompt_indices": [entry["idx"] for entry in chunk],
            }
        )
    return groups


def print_pooled_cos_stats(stats: Dict[str, Optional[float]], label: str) -> None:
    print(f"{label} over {stats['count']} prompts:")
    print(f"  min    = {stats['min']:.6f}")
    print(f"  max    = {stats['max']:.6f}")
    print(f"  mean   = {stats['mean']:.6f}")
    print(f"  median = {stats['median']:.6f}")


def print_pooled_cos_group_stats(groups: List[Dict], group_size: int = 20) -> None:
    print("=" * 60)
    print(f"pooled_cos stats by groups of {group_size} prompts:")
    for group in groups:
        stats = group["stats"]
        print(
            f"  [{group['range']}] "
            f"min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}"
        )


def save_pooled_cosine_json(
    prompt_results: List[Dict],
    cos_values: List[float],
    args,
    json_path: Optional[str] = None,
    group_size: int = 20,
) -> str:
    stats = compute_pooled_cos_stats(cos_values)
    groups = build_pooled_cos_groups(prompt_results, group_size=group_size)

    print("=" * 60)
    print_pooled_cos_stats(stats, "pooled_cos stats")
    print_pooled_cos_group_stats(groups, group_size=group_size)

    json_path = json_path or args.pooled_cosine_json or os.path.join(
        args.results_dir, "pooled_cosine_results.json"
    )
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    json_payload = {
        "task": args.task,
        "vector_type": args.vector_type,
        "pooled_cosine_score_mode": args.pooled_cosine_score_mode,
        "pooled_cosine_clip": args.pooled_cosine_clip,
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "group_size": group_size,
        "num_prompts": len(prompt_results),
        "stats": stats,
        "groups": groups,
        "prompts": prompt_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Apply SD3.5 dual-branch activation steering.")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, default="data/captions.txt")
    parser.add_argument("--seeds_path", type=str, default=None)
    parser.add_argument("--vector_type", type=str, default="diff")
    parser.add_argument("--num_prompts", type=int, default=500)
    parser.add_argument('--remove_prompt', type=str, default='cyberpunk style')
    parser.add_argument("--task", type=str, default="add concept", choices=["add concept", "remove", "nudity"])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--strength", type=float, default=20.0, help="Text branch strength")
    parser.add_argument("--strength_img", type=float, default=20.0, help="Image branch strength")
    parser.add_argument("--steer_txt", action="store_true", help="Also steer SD3 text encoder embeddings")
    parser.add_argument("--strength_txt", type=float, default=2.0, help="Text encoder steering strength")
    parser.add_argument("--txt_vector_path", type=str, default=None, help="Optional explicit text steering vector path")
    parser.add_argument(
        "--use_pooled_cosine_score",
        action="store_true",
        help="Multiply transformer steering by a pooled text-vector cosine coefficient.",
    )
    parser.add_argument(
        "--pooled_cosine_score_mode",
        type=str,
        default="similarity",
        choices=["similarity", "distance"],
        help="Coefficient from pooled vectors: positive cosine similarity or cosine distance.",
    )
    parser.add_argument(
        "--pooled_cosine_clip",
        type=float,
        default=0.3,
        help="Upper bound for pooled cosine multiplier used in transformer steering.",
    )
    parser.add_argument(
        "--test_pooled_cosine",
        action="store_true",
        help="Only compute pooled cosine between each prompt and text steering vector; skip image generation.",
    )
    parser.add_argument(
        "--pooled_cosine_json",
        type=str,
        default=None,
        help="Path to save pooled cosine results as JSON (default: <results_dir>/pooled_cosine_results.json).",
    )
    parser.add_argument(
        "--pooled_cosine_group_size",
        type=int,
        default=20,
        help="Group size for pooled_cos min/max/mean statistics.",
    )
    parser.add_argument("--steering_type", type=str, default="mean", choices=["mean", "separate"])
    parser.add_argument("--steering_mode", type=str, default="unconditional", choices=["unconditional", "cast"])
    parser.add_argument("--condition_vector_path", type=str, default=None)
    parser.add_argument("--cast_threshold", type=float, default=0.0)
    parser.add_argument("--cast_threshold_path", type=str, default=None)
    parser.add_argument("--cast_comparator", type=str, default="smaller", choices=["smaller", "larger"])
    parser.add_argument("--cast_layer", type=int, default=-1)
    parser.add_argument("--injection_point", type=str, default="block", choices=["attn", "block", "ff"])
    parser.add_argument("--steer_batch_slice", type=str, default="conditional", choices=["conditional", "all"])
    parser.add_argument("--use_ssim_mask", action="store_true")
    parser.add_argument("--top_k_percent", type=float, default=0.01)
    parser.add_argument("--use_cls", action="store_true")
    parser.add_argument("--min_signal_threshold", type=float, default=0.5)
    parser.add_argument("--block_steering", type=str, default="all")
    parser.add_argument("--t_steering", type=str, default="all")
    parser.add_argument("--quantile_level", type=float, default=0.5)
    parser.add_argument("--inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results_sd35_steered")
    parser.add_argument("--structure", type=float, default=0.5)
    parser.add_argument("--cls_min", type=float, default=21.0)
    parser.add_argument("--cls_type", type=str, default="tanh")
    args = parser.parse_args()

    if args.block_steering != "all":
        args.block_steering = [int(x) for x in args.block_steering.split(",")]
    if args.t_steering != "all":
        args.t_steering = [int(x) for x in args.t_steering.split(",")]

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="balanced",
        use_safetensors=True,
    )

    vector_path = find_vector_file(args.data_dir, args.vector_type)
    vector = torch.load(vector_path, map_location="cpu")
    print(f"Loaded: {vector_path}")

    txt_vector_path = find_text_vector_file(
        args.data_dir,
        vector_path,
        args.vector_type,
        explicit_path=args.txt_vector_path,
    )
    vector_txt_for_scale = None
    if txt_vector_path:
        vector_txt_for_scale = torch.load(txt_vector_path, map_location="cpu")
        print(f"Text pooled vector: {txt_vector_path}")

    vector_txt = vector_txt_for_scale if args.steer_txt else None
    if args.steer_txt and vector_txt is None:
        print("WARNING: --steer_txt set but no text vector found. Use --txt_vector_path to pass it explicitly.")

    txt_steering = {
        "name": "txt_0",
        "vector": vector_txt,
        "strength": args.strength_txt,
        "task": args.task,
        "use_pooled_cosine_score": args.use_pooled_cosine_score,
        "pooled_cosine_score_mode": args.pooled_cosine_score_mode,
        "pooled_cosine_clip": args.pooled_cosine_clip,
    }

    prompts, seeds = load_prompts_and_seeds(args)
    prompts = prompts[:]
    seeds = seeds[:len(prompts)]

    if args.test_pooled_cosine:
        if vector_txt_for_scale is None:
            raise FileNotFoundError(
                f"--test_pooled_cosine requires a text steering vector (*_text_{args.vector_type}.pt) in {args.data_dir}"
            )
        print(
            f"Testing pooled cosine for {len(prompts)} prompts "
            f"(mode={args.pooled_cosine_score_mode}, task={args.task})"
        )
        prompt_results, cos_values = collect_pooled_cosine_results(
            pipe, prompts, vector_txt_for_scale, args
        )
        save_pooled_cosine_json(
            prompt_results,
            cos_values,
            args,
            group_size=args.pooled_cosine_group_size,
        )
        print("Done (no images generated).")
        raise SystemExit(0)

    os.makedirs(os.path.join(args.results_dir, "steered"), exist_ok=True)
    names = []
    for idx, prompt in enumerate(prompts):
        # if args.task == 'remove':
        #     prompt = prompt + " " + args.remove_prompt
        sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        suffix = f"s_{args.strength}_simg_{args.strength_img}_v_{args.vector_type}_{args.injection_point}"
        print(f"idx: {idx}", prompt)
        # if idx < 3750:
        #     continue
        
        if args.steer_txt:
            suffix += f"_stxt_{args.strength_txt}"
        if args.use_pooled_cosine_score:
            suffix += f"_pcos_{args.pooled_cosine_score_mode}_clip_{args.pooled_cosine_clip}"
        if args.steering_mode == "cast":
            suffix += f"_cast_l{args.cast_layer}"
        out_path = os.path.join(args.results_dir, "steered", f"{idx:04d}_{sanitized}_{suffix}.png")
        if os.path.exists(out_path):
            names.append(f"{idx:04d}_{sanitized}_{suffix}.png")
            continue

        args._current_batch_size = 1
        pooled_scale = 1.0
        if args.use_pooled_cosine_score:
            if vector_txt_for_scale is None:
                raise FileNotFoundError(
                    f"--use_pooled_cosine_score requires a text steering vector (*_text_{args.vector_type}.pt) in {args.data_dir}"
                )
            pooled_scale = compute_pooled_cos_for_prompt(pipe, prompt, vector_txt_for_scale, args)
            pooled_scale = float(np.clip(pooled_scale, 0.0, args.pooled_cosine_clip))
            print(
                f"Pooled cosine scale for block injection: "
                f"{pooled_scale:.4f} (clip={args.pooled_cosine_clip})"
            )

        hook_state, remove_hooks = apply_attention_steering_sd35(pipe, args, vector, pooled_scale=pooled_scale)
        generator = torch.Generator().manual_seed(int(seeds[idx]))

        def _on_step_end(_pipe, i, _t, callback_kwargs):
            hook_state.update({"step": i + 1})
            return callback_kwargs

        try:
            images = pipe(
                prompt,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                width=args.width,
                height=args.height,
                generator=generator,
                structure_strength=args.structure,
                callback_on_step_end=_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                txt_steering=txt_steering,
            ).images
            images[0].save(out_path)
            if len(images) > 1:
                os.makedirs(os.path.join(args.results_dir, "origin"), exist_ok=True)
                images[1].save(os.path.join(args.results_dir, "origin", f"{idx:04d}_orig.png"))
            print(f"Saved {out_path}")
        finally:
            remove_hooks()
        if idx > 5000:
            break

        # if idx >= 10:
        #     break
    if args.use_pooled_cosine_score and vector_txt_for_scale is not None:
        prompt_results, cos_values = collect_pooled_cosine_results(
            pipe, prompts, vector_txt_for_scale, args
        )
        save_pooled_cosine_json(
            prompt_results,
            cos_values,
            args,
            group_size=args.pooled_cosine_group_size,
        )
        

    print(len(names))
    
    print('==============================================')
    torch.save(names, os.path.join("named_prompts_sd3_6.pt"))
        #assert False


if __name__ == "__main__":
    main()
