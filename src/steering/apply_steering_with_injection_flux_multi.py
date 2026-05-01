"""
Multi-vector attention + optional per-vector text steering for FLUX / SD3.

Configuration: pass --steering_config pointing to a JSON file. Example:

{
  "entries": [
    {
      "name": "concept_a",
      "attn_vector": "path/to/vectors_a_subspace_act_v1.pt",
      "data_dir": "path/to/svm_scores_for_a",
      "strength": 10.0,
      "use_cls": true,
      "txt_vector": "path/to/vectors_a_text_diff.pt",
      "strength_txt": 6.0,
      "task": null
    },
    {
      "name": "concept_b",
      "attn_vector": "/abs/path/to/b_diff.pt",
      "strength": 5.0,
      "use_cls": false
    }
  ]
}

Paths may be absolute or relative to --data_dir. Per-entry "task" overrides the
global --task when computing that bundle's classifier / steering direction;
omit or null to use --task.

Text steering: pass --steer_txt. The pipeline receives txt_steering["vectors"]
with one item per entry that has "txt_vector" set (see flux.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))

from src.models.flux import FluxPipeline
from src.models.sd_3 import StableDiffusion3Pipeline
from src.steering.apply_steering_with_injection_flux import (
    MODEL_CONFIGS,
    SteeringEngine,
    load_steering_data,
)
from src.utils.utils import calculate_cls_score
from datasets import load_dataset


@dataclass
class SteeringBundle:
    name: str
    vector: Dict
    models: Optional[Dict]
    scores_all: Optional[torch.Tensor]
    strength: float
    use_cls: bool
    task: str
    txt_vector: Optional[Dict] = None
    strength_txt: float = 1.0


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _default_svm_paths(data_dir: str) -> Tuple[str, str, str]:
    svm = os.path.join(data_dir, "base_0.85_20_svm_models.pt")
    scr = os.path.join(data_dir, "base_0.85_20_scores.pt")
    tok = os.path.join(data_dir, "base_0.85_20_tokens.pt")
    return svm, scr, tok


def load_steering_bundles(
    config_path: str,
    global_data_dir: str,
    global_task: str,
    global_strength: float,
    use_cls_default: bool,
    strength_txt_default: float,
    quantile_level: float,
) -> List[SteeringBundle]:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    entries = cfg.get("entries")
    if not entries:
        raise ValueError(f"Config {config_path} must contain a non-empty 'entries' list.")

    bundles: List[SteeringBundle] = []
    for raw in entries:
        name = raw.get("name") or f"bundle_{len(bundles)}"
        data_dir = resolve_path(raw.get("data_dir"), global_data_dir) or global_data_dir
        attn_path = resolve_path(raw.get("attn_vector"), global_data_dir)
        if not attn_path or not os.path.isfile(attn_path):
            raise FileNotFoundError(f"[{name}] attn_vector missing or not a file: {attn_path}")

        svm = raw.get("svm_models")
        scr = raw.get("scores")
        tok = raw.get("tokens_best")
        if svm is None or scr is None or tok is None:
            ds, ss, tt = _default_svm_paths(data_dir)
            svm = svm or ds
            scr = scr or ss
            tok = tok or tt
        svm = resolve_path(svm, global_data_dir)
        scr = resolve_path(scr, global_data_dir)
        tok = resolve_path(tok, global_data_dir)

        models, _tokens_best, scores_all, _q = load_steering_data(
            svm, scr, tok, quantile_level
        )
        strength = float(raw.get("strength", global_strength))
        use_cls = raw.get("use_cls", use_cls_default)
        if isinstance(use_cls, str):
            use_cls = use_cls.lower() in ("1", "true", "yes")
        task = raw.get("task") or global_task
        txt_path = resolve_path(raw.get("txt_vector"), global_data_dir)
        txt_vec = None
        if txt_path:
            if not os.path.isfile(txt_path):
                raise FileNotFoundError(f"[{name}] txt_vector not found: {txt_path}")
            txt_vec = torch.load(txt_path, weights_only=False)
        s_txt = float(raw.get("strength_txt", strength_txt_default))

        bundles.append(
            SteeringBundle(
                name=name,
                vector=torch.load(attn_path, weights_only=False),
                models=models,
                scores_all=scores_all,
                strength=strength,
                use_cls=bool(use_cls),
                task=task,
                txt_vector=txt_vec,
                strength_txt=s_txt,
            )
        )
    return bundles


def apply_attention_steering_multi(
    pipe,
    args,
    bundles: List[SteeringBundle],
    eigen_info: Optional[Any] = None,
    verbose: bool = False,
):
    m_key = "flux" if "flux" in args.model_name.lower() else "sd3"
    cfg = MODEL_CONFIGS[m_key]
    state = {"step": 0}
    hook_handles = []

    def steering_hook(layer_idx: int):
        def hook(module, input, output):
            step = state["step"]
            if layer_idx == cfg["last_layer"]:
                state["step"] += 1

            if verbose:
                print(step, layer_idx)

            if args.block_steering != "all" and layer_idx not in args.block_steering:
                return output
            if args.t_steering != "all" and step not in args.t_steering:
                return output

            if len(output) != 2:
                return output

            act_tuple = list(output)
            hidden_states = act_tuple[cfg["out_idx"]]
            current = hidden_states[cfg["inner_idx"]].clone()

            for bundle in bundles:
                if step not in bundle.vector or f"layer_{layer_idx}" not in bundle.vector[step]:
                    continue

                score_val = torch.ones((1, 1), device=current.device, dtype=current.dtype)
                current_signal = 1.0
                if bundle.scores_all is not None:
                    layer_scores = bundle.scores_all[:, step, layer_idx]
                    current_signal = layer_scores.float()

                if bundle.use_cls and bundle.models:
                    ensemble = bundle.models.get(step, {}).get(f"layer_{layer_idx}")
                    if ensemble:
                        mean_act = current.mean(0, keepdim=True)
                        mean_act = mean_act / (mean_act.norm(dim=-1, keepdim=True) + 1e-6)
                        votes = [
                            calculate_cls_score(
                                mean_act.cpu().float(),
                                args.cls_min,
                                m,
                                args.cls_type,
                                task=bundle.task,
                                use_distance=False,
                            )[0]
                            if current_signal[i] > args.min_signal_threshold
                            else 1.0
                            for i, m in enumerate(ensemble)
                        ]
                        score_val = torch.tensor(votes).to(current.device, current.dtype)
                        if args.vector_type == "diff" or args.steering_type == "mean":
                            score_val = torch.mean(score_val, dim=0, keepdim=True)
                else:
                    score_val = torch.ones((1, 1), device=current.device, dtype=current.dtype)

                if verbose:
                    print(bundle.name, score_val)

                if args.steering_type == "mean":
                    steering_vec = (
                        bundle.vector[step][f"layer_{layer_idx}"].mean(0, keepdim=True).to(current.device)
                    )
                else:
                    steering_vec = bundle.vector[step][f"layer_{layer_idx}"].to(
                        current.device, current.dtype
                    )

                args_eff = copy.copy(args)
                args_eff.strength = bundle.strength
                args_eff.task = bundle.task
                current = SteeringEngine.apply_steering(
                    current, steering_vec, args_eff, score_val
                )

            act_tuple[cfg["out_idx"]][cfg["inner_idx"]] = current
            return tuple(act_tuple)

        return hook

    layer_id = 0
    for name, module in pipe.transformer.named_modules():
        if name.endswith("attn"):
            hook_handles.append(module.register_forward_hook(steering_hook(layer_id)))
            layer_id += 1

    return state, lambda: [h.remove() for h in hook_handles]


def build_txt_steering(
    steer_txt: bool,
    bundles: List[SteeringBundle],
    default_task: str,
    default_strength_txt: float,
) -> Dict[str, Any]:
    if not steer_txt:
        return {"vector": None, "strength": default_strength_txt, "task": default_task}
    vec_entries = []
    for b in bundles:
        if b.txt_vector is None:
            continue
        vec_entries.append(
            {
                "vector": b.txt_vector,
                "strength": b.strength_txt,
                "task": b.task,
            }
        )
    if not vec_entries:
        return {"vector": None, "strength": default_strength_txt, "task": default_task}
    return {"vectors": vec_entries, "strength": default_strength_txt, "task": default_task}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base directory for resolving relative paths in the JSON config.",
    )
    parser.add_argument(
        "--steering_config",
        type=str,
        required=True,
        help="JSON file with an 'entries' list (see module docstring).",
    )
    parser.add_argument("--prompts_path", type=str, default="data/captions.txt")
    parser.add_argument("--vector_type", type=str, default="diff")
    parser.add_argument("--task", type=str, default="add concept", choices=["add concept", "remove", "nudity"])
    parser.add_argument("--remove_prompt", type=str, default="cyberpunk style")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    parser.add_argument("--strength", type=float, default=20.0, help="Default strength when an entry omits 'strength'.")
    parser.add_argument("--steering_type", type=str, default="mean")
    parser.add_argument("--use_ssim_mask", action="store_true")
    parser.add_argument("--top_k_percent", type=float, default=0.01)
    parser.add_argument("--use_cls", action="store_true", help="Default use_cls for entries that omit it.")
    parser.add_argument("--min_signal_threshold", type=float, default=0.5)

    parser.add_argument("--block_steering", type=str, default="all")
    parser.add_argument("--t_steering", type=str, default="all")
    parser.add_argument("--quantile_level", type=float, default=0.5)

    parser.add_argument("--steer_txt", action="store_true")
    parser.add_argument("--strength_txt", type=float, default=2.0, help="Default text strength for entries that omit 'strength_txt'.")
    parser.add_argument("--inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results_steered_multi")
    parser.add_argument("--structure", type=float, default=0.5)
    parser.add_argument("--cls_min", type=float, default=21.0)
    parser.add_argument("--cls_type", type=str, default="tanh")
    parser.add_argument("--verbose_hooks", action="store_true")

    args = parser.parse_args()

    if args.block_steering != "all":
        args.block_steering = [int(x) for x in args.block_steering.split(",")]
    if args.t_steering != "all":
        args.t_steering = [int(x) for x in args.t_steering.split(",")]

    bundles = load_steering_bundles(
        args.steering_config,
        args.data_dir,
        args.task,
        args.strength,
        args.use_cls,
        args.strength_txt,
        args.quantile_level,
    )
    txt_steering = build_txt_steering(args.steer_txt, bundles, args.task, args.strength_txt)

    is_flux = "flux" in args.model_name.lower()
    pipe = (FluxPipeline if is_flux else StableDiffusion3Pipeline).from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if is_flux else torch.float16,
        device_map="balanced",
        use_safetensors=True,
    )

    try:
        with open(args.prompts_path, "r") as f:
            coco_prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        coco_prompts = ["A high quality photo"]

    if args.task == "nudity":
        dataset = load_dataset("AIML-TUDA/i2p", split="train")
        coco_prompts = [sample["prompt"] for sample in dataset]
        coco_seeds = [int(sample["sd_seed"]) for sample in dataset]
    else:
        coco_seeds = [int(args.seed) for _ in range(len(coco_prompts))]

    names_tag = "_".join(b.name for b in bundles[:3])
    if len(bundles) > 3:
        names_tag += f"_plus{len(bundles) - 3}"

    for idx, prompt in enumerate(coco_prompts):
        if args.task == "remove":
            prompt = prompt + " " + args.remove_prompt

        sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        suffix = f"multi_{names_tag}_mask_{args.use_ssim_mask}_vt_{args.vector_type}"

        out_path = os.path.join(args.results_dir, "steered", f"{idx:02d}_{sanitized}_{suffix}.png")
        if os.path.exists(out_path):
            continue

        seed = int(coco_seeds[idx])
        print(f"Processing {idx + 1}/{len(coco_prompts)}: {prompt}", seed, args.inference_steps, args.guidance_scale)

        hook_state, remove_hooks = apply_attention_steering_multi(
            pipe, args, bundles, verbose=args.verbose_hooks
        )
        generator = torch.Generator("cpu").manual_seed(seed)

        images = pipe(
            prompt,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=generator,
            structure_strength=args.structure,
            callback=lambda step, **k: hook_state.update({"step": step}),
            callback_steps=1,
            txt_steering=txt_steering,
        ).images

        os.makedirs(os.path.join(args.results_dir, "steered"), exist_ok=True)
        try:
            images[0].save(out_path)
        except Exception as exc:
            print(f"Failed to save steered image for index {idx}: {exc}")
        if len(images) > 1:
            os.makedirs(os.path.join(args.results_dir, "origin"), exist_ok=True)
            images[1].save(os.path.join(args.results_dir, "origin", f"{idx:02d}_orig.png"))

        remove_hooks()
