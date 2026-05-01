"""
Standalone inference script for text pooled steering in low-d subspace.

This script does NOT inject transformer hooks (unlike subspace_act apply).
It only steers text encoder projections via txt_steering vector:
  - classic text diff (legacy)
  - subspace mean-diff (pooled_steering_mode='subspace_mean')
  - Monge map in subspace (pooled_steering_mode='monge')
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))

from src.models.flux import FluxPipeline
from src.models.sd_3 import StableDiffusion3Pipeline


def _load_text_vector(path: str, key: str | None) -> Any:
    loaded = torch.load(path, map_location="cpu")
    if key is None:
        return loaded
    if not isinstance(loaded, dict):
        raise TypeError("--text_vector_key was passed, but loaded object is not a dict.")
    if key not in loaded:
        keys = list(loaded.keys())
        raise KeyError(
            f"--text_vector_key={key!r} is missing in {path}. "
            f"Available keys ({len(keys)}): {keys[:50]}" + (" ..." if len(keys) > 50 else "")
        )
    return loaded[key]


def _load_prompts_and_seeds(args):
    try:
        with open(args.prompts_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        prompts = ["A high quality photo"]

    if args.task == "nudity":
        dataset = load_dataset("AIML-TUDA/i2p", split="train")
        prompts = [sample["prompt"] for sample in dataset]
        seeds = [int(sample["sd_seed"]) for sample in dataset]
    else:
        seeds = [int(args.seed) for _ in range(len(prompts))]

    if args.num_prompts is not None and args.num_prompts > 0:
        prompts = prompts[: args.num_prompts]
        seeds = seeds[: args.num_prompts]

    return prompts, seeds


def main():
    parser = argparse.ArgumentParser(
        description="Apply text pooled OT/subspace steering without transformer hooks."
    )
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--text_vector_path", type=str, required=True)
    parser.add_argument(
        "--text_vector_key",
        type=str,
        default=None,
        help="Optional key if text_vector_path points to a stats/bundle dict.",
    )
    parser.add_argument(
        "--legacy_txt_steering",
        action="store_true",
        help="Force classic text steering path even if vector has advanced mode.",
    )

    parser.add_argument("--prompts_path", type=str, default="data/captions.txt")
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--task", type=str, default="add concept", choices=["add concept", "remove", "nudity"])
    parser.add_argument("--remove_prompt", type=str, default="cyberpunk style")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results_txt_pooled_ot")

    parser.add_argument("--strength_txt", type=float, default=6.0)
    parser.add_argument("--structure", type=float, default=0.5)

    args = parser.parse_args()

    is_flux = "flux" in args.model_name.lower()
    pipe = (FluxPipeline if is_flux else StableDiffusion3Pipeline).from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if is_flux else torch.float16,
        device_map="balanced",
        use_safetensors=True,
    )

    text_vec_path = os.path.abspath(args.text_vector_path)
    if not os.path.isfile(text_vec_path):
        raise FileNotFoundError(f"--text_vector_path not found: {text_vec_path}")

    vector_txt = _load_text_vector(text_vec_path, args.text_vector_key)
    txt_steering = {
        "vector": vector_txt,
        "strength": args.strength_txt,
        "task": args.task,
    }
    if args.legacy_txt_steering:
        txt_steering["legacy_txt_steering"] = True

    prompts, seeds = _load_prompts_and_seeds(args)
    os.makedirs(os.path.join(args.results_dir, "steered"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "origin"), exist_ok=True)

    vector_tag = args.text_vector_key if args.text_vector_key else Path(text_vec_path).stem
    vector_tag = vector_tag.replace("/", "_").replace(" ", "_")

    for idx, prompt in enumerate(prompts):
        p = prompt
        if args.task == "remove":
            p = p + " " + args.remove_prompt

        sanitized = p.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        suffix = f"stxt_{args.strength_txt}_v_{vector_tag}"
        out_path = os.path.join(args.results_dir, "steered", f"{idx:02d}_{sanitized}_{suffix}.png")
        if os.path.exists(out_path):
            continue

        seed = int(seeds[idx])
        generator = torch.Generator().manual_seed(seed)
        print(f"{idx + 1}/{len(prompts)}: {p[:70]}... seed={seed}")

        images = pipe(
            p,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=generator,
            structure_strength=args.structure,
            txt_steering=txt_steering,
        ).images

        try:
            images[0].save(out_path)
        except Exception as exc:
            print(f"Save failed idx {idx}: {exc}")

        if len(images) > 1:
            images[1].save(os.path.join(args.results_dir, "origin", f"{idx:02d}_orig.png"))

    print("Done.")


if __name__ == "__main__":
    main()
