"""
Multi-concept SHIFT for FLUX: several steering fields applied together, each with
its own SVM scale and (optionally) its own cosine gate.

Configuration: pass --steering_config pointing to a JSON file. Example:

{
  "entries": [
    {
      "name": "hate",
      "vector": "hate/base_0.85_135_diff.pt",
      "data_dir": "hate",
      "strength": 225.0,
      "use_cls": true,
      "txt_vector": "hate/base_0.85_135_text_diff.pt",
      "strength_txt": 6.0
    },
    {
      "name": "violence",
      "vector": "violence/base_0.85_135_diff.pt",
      "data_dir": "violence",
      "txt_vector": "violence/base_0.85_135_text_diff.pt"
    }
  ]
}

Paths may be absolute or relative to --data_dir. "vector" (alias "attn_vector") is the
DiT steering field; "data_dir" holds that concept's classifier (*_svm_models.pt, or set
"svm_prefix"). Per-entry "task", "strength", "strength_txt", "use_cls" override the
global flags. With --use_pooled_cosine_score every entry needs "txt_vector": its gate
c_k = clip(cos, 0, delta) is computed per prompt, so a field only activates on prompts
relevant to its concept.

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
from typing import Any, Dict, List, Optional

import torch

STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))

from src.models.flux import FluxPipeline
from src.steering.apply_steering_with_injection_flux import (
    MODEL_CONFIGS,
    SteeringEngine,
    classifier_scale,
    find_svm_files,
    load_steering_data,
    split_vector,
)
from src.utils.utils import pooled_cosine_gate
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
        vec_path = resolve_path(raw.get("vector") or raw.get("attn_vector"), global_data_dir)
        if not vec_path or not os.path.isfile(vec_path):
            raise FileNotFoundError(f"[{name}] vector missing or not a file: {vec_path}")

        svm, scr, tok = find_svm_files(data_dir, raw.get("svm_prefix"))
        svm = resolve_path(raw.get("svm_models"), global_data_dir) or svm
        scr = resolve_path(raw.get("scores"), global_data_dir) or scr
        tok = resolve_path(raw.get("tokens_best"), global_data_dir) or tok

        models, _tokens_best, scores_all, _q = load_steering_data(
            svm, scr, tok, quantile_level
        )
        use_cls = raw.get("use_cls", use_cls_default)
        if isinstance(use_cls, str):
            use_cls = use_cls.lower() in ("1", "true", "yes")
        if use_cls and models is None:
            print(f"WARNING: [{name}] use_cls set but no classifier found at {svm}")
        txt_path = resolve_path(raw.get("txt_vector"), global_data_dir)
        txt_vec = None
        if txt_path:
            if not os.path.isfile(txt_path):
                raise FileNotFoundError(f"[{name}] txt_vector not found: {txt_path}")
            txt_vec = torch.load(txt_path, weights_only=False)

        bundles.append(
            SteeringBundle(
                name=name,
                vector=split_vector(torch.load(vec_path, weights_only=False))[1],
                models=models,
                scores_all=scores_all,
                strength=float(raw.get("strength", global_strength)),
                use_cls=bool(use_cls),
                task=raw.get("task") or global_task,
                txt_vector=txt_vec,
                strength_txt=float(raw.get("strength_txt", strength_txt_default)),
            )
        )
    return bundles


def apply_attention_steering_multi(
    pipe,
    args,
    bundles: List[SteeringBundle],
    gates: Dict[str, float],
    verbose: bool = False,
):
    """
    Hooks the double-stream blocks and applies every bundle's field in turn to the
    text stream (post-block state for --injection_point block, joint-attention
    output for attn). `gates` maps bundle name -> cosine gate for this prompt.
    """
    cfg = MODEL_CONFIGS["flux"]
    state = {"step": 0, "t": 0}  # step: vector index, t: denoising step
    hook_handles = []

    def steer(current: torch.Tensor, step: int, layer_idx: int) -> torch.Tensor:
        layer_key = f"layer_{layer_idx}"
        for bundle in bundles:
            gate = gates.get(bundle.name, 1.0)
            if gate == 0.0 or layer_key not in bundle.vector.get(step, {}):
                continue

            args_eff = copy.copy(args)
            args_eff.strength = bundle.strength
            args_eff.task = bundle.task
            args_eff.use_cls = bundle.use_cls
            score_val = classifier_scale(
                step, layer_idx, current, bundle.models, bundle.scores_all, args_eff
            ) * gate

            if verbose:
                print(bundle.name, step, layer_idx, score_val.flatten().tolist())

            sv_raw = bundle.vector[step][layer_key]
            if args.steering_type == "mean":
                steering_vec = sv_raw.mean(0, keepdim=True).to(current.device)
            else:
                steering_vec = sv_raw.to(current.device, current.dtype)
            current = SteeringEngine.apply_steering(current, steering_vec, args_eff, score_val)
        return current

    def steering_hook(layer_idx: int):
        def hook(module, input, output):
            if args.block_steering != "all" and layer_idx not in args.block_steering:
                return output
            if args.t_steering != "all" and state["t"] not in args.t_steering:
                return output
            if not (isinstance(output, tuple) and len(output) == 2):
                return output

            if args.injection_point == "block":
                # FluxTransformerBlock returns (txt, img)
                txt, img = output
                return (steer(txt.clone(), state["step"], layer_idx), img)

            # joint attention returns (img_attn_out, txt_attn_out)
            act_tuple = list(output)
            txt = act_tuple[cfg["out_idx"]]
            act_tuple[cfg["out_idx"]][cfg["inner_idx"]] = steer(
                txt[cfg["inner_idx"]].clone(), state["step"], layer_idx
            )
            return tuple(act_tuple)

        return hook

    for layer_id, block in enumerate(pipe.transformer.transformer_blocks):
        module = block if args.injection_point == "block" else block.attn
        hook_handles.append(module.register_forward_hook(steering_hook(layer_id)))

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
    parser.add_argument("--task", type=str, default="remove", choices=["add concept", "remove", "nudity"],
                        help="'nudity' evaluates on the I2P benchmark with its seeds.")
    parser.add_argument("--prompt_suffix", type=str, default="", help="Appended to every prompt (e.g. a concept name).")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    parser.add_argument("--strength", type=float, default=20.0, help="Default strength when an entry omits 'strength'.")
    parser.add_argument("--steering_type", type=str, default="separate")
    parser.add_argument("--injection_point", type=str, default="block", choices=["block", "attn"])
    parser.add_argument("--use_ssim_mask", action="store_true")
    parser.add_argument("--top_k_percent", type=float, default=0.01)
    parser.add_argument("--use_cls", action="store_true", help="Default use_cls for entries that omit it.")
    parser.add_argument("--min_signal_threshold", type=float, default=0.5)

    parser.add_argument("--use_pooled_cosine_score", action="store_true",
                        help="Per-concept cosine gate on the DiT route (every entry needs txt_vector).")
    parser.add_argument("--pooled_cosine_clip", type=float, default=0.15, help="Gate upper bound delta.")
    parser.add_argument("--pooled_cosine_threshold", type=float, default=0.0, help="Zero gates below this value.")

    parser.add_argument("--block_steering", type=str, default="all")
    parser.add_argument("--t_steering", type=str, default="all")
    parser.add_argument("--per_step_vectors", action="store_true",
                        help="Use the vector of each denoising step instead of reusing the step-0 vector. "
                             "Steps without a stored vector are not steered.")
    parser.add_argument("--quantile_level", type=float, default=0.5)

    parser.add_argument("--steer_txt", action="store_true")
    parser.add_argument("--strength_txt", type=float, default=2.0, help="Default text strength for entries that omit 'strength_txt'.")
    parser.add_argument("--inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds_path", type=str, default=None, help="One seed per prompt. Default: --seed for all.")
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
    if args.use_pooled_cosine_score:
        missing = [b.name for b in bundles if b.txt_vector is None]
        if missing:
            parser.error(f"--use_pooled_cosine_score needs txt_vector for: {missing}")
        if any(b.task == "add concept" for b in bundles):
            parser.error("--use_pooled_cosine_score is defined for suppression entries only")
    cls_add = [b.name for b in bundles if b.use_cls and b.task == "add concept"]
    if cls_add:
        parser.error(f"use_cls is defined for suppression entries only: {cls_add}")
    txt_steering = build_txt_steering(args.steer_txt, bundles, args.task, args.strength_txt)

    pipe = FluxPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
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
    elif args.seeds_path:
        with open(args.seeds_path, "r") as f:
            coco_seeds = [line.strip() for line in f if line.strip()]
    else:
        coco_seeds = [int(args.seed) for _ in range(len(coco_prompts))]

    names_tag = "_".join(b.name for b in bundles[:3])
    if len(bundles) > 3:
        names_tag += f"_plus{len(bundles) - 3}"

    for idx, prompt in enumerate(coco_prompts):
        if args.prompt_suffix:
            prompt = f"{prompt} {args.prompt_suffix}"
        sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        suffix = f"multi_{names_tag}_mask_{args.use_ssim_mask}_vt_{args.vector_type}"

        out_path = os.path.join(args.results_dir, "steered", f"{idx:02d}_{sanitized}_{suffix}.png")
        if os.path.exists(out_path):
            continue

        gates = {}
        if args.use_pooled_cosine_score:
            with torch.no_grad():
                pooled = pipe._get_clip_prompt_embeds(prompt=prompt, device=pipe._execution_device)
            gates = {
                b.name: pooled_cosine_gate(
                    pipe, prompt, b.txt_vector, pooled=pooled,
                    clip_max=args.pooled_cosine_clip, threshold=args.pooled_cosine_threshold,
                )
                for b in bundles
            }

        seed = int(coco_seeds[idx])
        print(f"Processing {idx + 1}/{len(coco_prompts)}: {prompt[:60]}... seed={seed} gates={gates}")

        hook_state, remove_hooks = apply_attention_steering_multi(
            pipe, args, bundles, gates, verbose=args.verbose_hooks
        )
        generator = torch.Generator("cpu").manual_seed(seed)

        def _on_step_end(_pipe, i, _t, callback_kwargs):
            hook_state.update({"step": i + 1 if args.per_step_vectors else 0, "t": i + 1})
            return callback_kwargs

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

        os.makedirs(os.path.join(args.results_dir, "steered"), exist_ok=True)
        try:
            images[0].save(out_path)
        except Exception as exc:
            print(f"Failed to save steered image for index {idx}: {exc}")
        if len(images) > 1:
            os.makedirs(os.path.join(args.results_dir, "origin"), exist_ok=True)
            images[1].save(os.path.join(args.results_dir, "origin", f"{idx:02d}_orig.png"))

        remove_hooks()
