"""
SD3.5 MM-DiT activation extraction.

Saves activations in the same dual-branch format used by the Flux extractor:
  vectors[step]["layer_N"] = {"img": tensor, "txt": tensor}

For SD3/SD3.5 JointTransformerBlock:
  - block output is (encoder_hidden_states, hidden_states) = (txt, img)
  - attn output is (hidden_states, encoder_hidden_states) = (img, txt)
  - ff is image branch, ff_context is text branch when present
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm

STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))

from src.models.sd_3 import StableDiffusion3Pipeline


def get_imagenet_classes(num: int = 50, prompt_path: str = "imagenet_classes_animals.txt") -> List[str]:
    try:
        with open(prompt_path, "r") as f:
            return [line.strip() for i, line in enumerate(f) if i < num and line.strip()]
    except FileNotFoundError:
        return [f"class_{i}" for i in range(num)]


def get_prompts_style(num=50, concept_pos="anime", concept_neg=None, prompt_path="imagenet_classes_animals.txt"):
    classes = get_imagenet_classes(num, prompt_path=prompt_path)
    pos = [f"{cls}, {concept_pos} style" for cls in classes]
    neg = [f"{cls}, {concept_neg} style" if concept_neg else cls for cls in classes]
    return pos, neg


def get_prompts_concrete(num=50, concept_pos="Snoopy", concept_neg=None, prompt_path="imagenet_classes_animals.txt"):
    classes = get_imagenet_classes(num, prompt_path=prompt_path)
    pos = [f"{cls} with {concept_pos}" for cls in classes]
    neg = [f"{cls} with {concept_neg}" if concept_neg else cls for cls in classes]
    return pos, neg


def get_prompts_switch(num=50, concept_pos="Snoopy", concept_neg=None, prompt_path="imagenet_classes_animals.txt"):
    classes = get_imagenet_classes(num, prompt_path=prompt_path)
    pos = [f"{concept_pos} {cls}" for cls in classes]
    neg = [f"{concept_neg} {cls}" if concept_neg else cls for cls in classes]
    return pos, neg


def get_prompts_human_related(num=100, concept_pos="nudity", concept_neg=None, prompt_path=None):
    people = [
        "a girl", "a boy", "two men", "two women", "two people", "a man", "a woman",
        "an old man", "an old woman", "boys", "girls", "man", "woman",
        "group of people", "a human",
    ]
    contexts = ["", "gloomy image", "zoomed in", "talking", "on a beach", "in a strange pose", "realism",
                "colorful background", "smiling"]
    prompts_pos, prompts_neg = [], []
    for person in people:
        for context in contexts:
            base = f"{person} {context}".strip()
            prompts_pos.append(f"{base}, {concept_pos}")
            prompts_neg.append(f"{base}, {concept_neg}" if concept_neg else base)
    return prompts_pos[:num], prompts_neg[:num]


PROMPT_FUNCS = {
    "style": get_prompts_style,
    "concrete": get_prompts_concrete,
    "switch": get_prompts_switch,
    "people": get_prompts_human_related,
}


class SD35HookManager:
    def __init__(
        self,
        extraction_point: str,
        save_blocks: int,
        save_timesteps: int,
        stream: str = "both",
        use_cfg: bool = False,
    ):
        self.extraction_point = extraction_point
        self.save_blocks = save_blocks
        self.save_timesteps = save_timesteps
        self.stream = stream
        self.use_cfg = use_cfg
        self.hooks_per_step = 1
        self.current_batch_size = 1
        self.data: Dict = {}
        self.reset_state()

    def reset_state(self, batch_size: Optional[int] = None):
        self.current_step = 0
        self.hooks_fired_this_step = 0
        if batch_size is not None:
            self.current_batch_size = batch_size

    def _advance_step_counter(self):
        self.hooks_fired_this_step += 1
        if self.hooks_fired_this_step >= self.hooks_per_step:
            self.current_step += 1
            self.hooks_fired_this_step = 0

    def _should_save(self, block_idx: int) -> bool:
        return block_idx < self.save_blocks and self.current_step < self.save_timesteps

    def _wants_branch(self, branch: str) -> bool:
        return self.stream == "both" or self.stream == branch

    def _select_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        batch = max(1, self.current_batch_size)
        if tensor.shape[0] >= 2 * batch:
            if self.use_cfg:
                return tensor[batch: 2 * batch]
            return tensor[:batch]
        return tensor

    @staticmethod
    def _first_tensor(output):
        if isinstance(output, tuple):
            return output[0]
        return output

    def _store(self, block_idx: int, branch: str, tensor: Optional[torch.Tensor]):
        if tensor is None or not self._wants_branch(branch) or not self._should_save(block_idx):
            return
        tensor = self._first_tensor(tensor)
        if not torch.is_tensor(tensor):
            return
        tensor = self._select_rows(tensor)
        step = self.current_step
        layer_key = f"layer_{block_idx}"
        self.data.setdefault(step, {}).setdefault(layer_key, {}).setdefault(branch, []).append(tensor.detach().cpu())

    def make_block_hook(self, block_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                txt, img = output[0], output[1]
                self._store(block_idx, "txt", txt)
                self._store(block_idx, "img", img)
            self._advance_step_counter()
        return hook_fn

    def make_attn_hook(self, block_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                img, txt = output[0], output[1]
                self._store(block_idx, "img", img)
                self._store(block_idx, "txt", txt)
            elif torch.is_tensor(output):
                self._store(block_idx, "img", output)
            self._advance_step_counter()
        return hook_fn

    def make_single_hook(self, block_idx: int, branch: str):
        def hook_fn(module, input, output):
            self._store(block_idx, branch, output)
            self._advance_step_counter()
        return hook_fn

    def aggregate(self) -> Dict:
        result = {}
        for step, layers in self.data.items():
            result[step] = {}
            for layer_key, branches in layers.items():
                result[step][layer_key] = {}
                for branch, tensor_list in branches.items():
                    if tensor_list:
                        result[step][layer_key][branch] = torch.cat(tensor_list, dim=0)
        return result


def register_hooks(pipe, manager: SD35HookManager, extraction_point: str, num_blocks: int) -> List:
    handles = []
    blocks = pipe.transformer.transformer_blocks
    for idx, block in enumerate(blocks[:num_blocks]):
        if extraction_point == "block":
            handles.append(block.register_forward_hook(manager.make_block_hook(idx)))
        elif extraction_point == "attn":
            handles.append(block.attn.register_forward_hook(manager.make_attn_hook(idx)))
        elif extraction_point == "norm1":
            handles.append(block.norm1.register_forward_hook(manager.make_single_hook(idx, "img")))
            if hasattr(block, "norm1_context") and block.norm1_context is not None:
                handles.append(block.norm1_context.register_forward_hook(manager.make_single_hook(idx, "txt")))
        elif extraction_point == "norm2":
            handles.append(block.norm2.register_forward_hook(manager.make_single_hook(idx, "img")))
            if hasattr(block, "norm2_context") and block.norm2_context is not None:
                handles.append(block.norm2_context.register_forward_hook(manager.make_single_hook(idx, "txt")))
        elif extraction_point == "ff":
            handles.append(block.ff.register_forward_hook(manager.make_single_hook(idx, "img")))
            if hasattr(block, "ff_context") and block.ff_context is not None:
                handles.append(block.ff_context.register_forward_hook(manager.make_single_hook(idx, "txt")))
        else:
            raise ValueError(f"Unknown extraction_point: {extraction_point}")

    manager.hooks_per_step = max(1, len(handles))
    return handles


def run_extraction(pipe, prompts, args) -> Tuple[Dict, List]:
    manager = SD35HookManager(
        extraction_point=args.extraction_point,
        save_blocks=args.num_layers,
        save_timesteps=args.save_timesteps,
        stream=args.token_stream,
        use_cfg=args.gs > 1.0,
    )
    handles = register_hooks(pipe, manager, args.extraction_point, args.num_layers)
    all_images = []

    try:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Extracting"):
            batch = prompts[i:i + args.batch_size]
            manager.reset_state(batch_size=len(batch))
            generators = [torch.Generator("cuda").manual_seed(42000 + i * 10 + j) for j in range(len(batch))]
            result = pipe(
                batch,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.gs,
                height=args.height,
                width=args.width,
                generator=generators,
                structure_strength=args.structure,
            )
            all_images.extend(result.images)
    finally:
        for handle in handles:
            handle.remove()

    vectors = manager.aggregate()
    for step in sorted(k for k in vectors if isinstance(k, int))[:1]:
        for layer_key in sorted(vectors[step]):
            shapes = {branch: tuple(t.shape) for branch, t in vectors[step][layer_key].items()}
            print(f"  step {step} {layer_key}: {shapes}")
    return vectors, all_images


def save_grid(images, filename):
    if not images:
        return
    nrow = max(1, int(len(images) ** 0.5))
    tensors = [ToTensor()(img) for img in images]
    grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
    img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    img.save(filename)


def main():
    parser = argparse.ArgumentParser(description="SD3.5 dual-branch activation extraction")
    parser.add_argument("--task", type=str, default="style", choices=["style", "concrete", "switch", "people"])
    parser.add_argument("--exp_type", type=str, default="3d")
    parser.add_argument("--pos_concept", type=str, default="picasso")
    parser.add_argument("--neg_concept", type=str, default="realistic")
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--prompt_path", type=str, default="imagenet_classes.txt")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--gs", type=float, default=4.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--structure", type=float, default=0.5)
    parser.add_argument("--extraction_point", type=str, default="block", choices=["block", "attn", "norm1", "norm2", "ff"])
    parser.add_argument("--token_stream", type=str, default="both", choices=["txt", "img", "both"])
    parser.add_argument("--num_layers", type=int, default=24, help="SD3.5 Medium has 24 transformer blocks")
    parser.add_argument("--save_timesteps", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="test_vectors_sd35/style")
    parser.add_argument("--save_image_dir", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading {args.model_name}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="balanced",
        use_safetensors=True,
    )

    prompt_func = PROMPT_FUNCS[args.task]
    pos_prompts, neg_prompts = prompt_func(
        num=args.num_prompts,
        concept_pos=args.pos_concept,
        concept_neg=args.neg_concept,
        prompt_path=args.prompt_path,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    f_template = (
        f"{args.exp_type}_{args.neg_concept}_gs_{args.gs}_"
        f"prompts_{args.num_prompts}_{{}}_{args.extraction_point}.pt"
    )
    h_latent = args.height // 16
    w_latent = args.width // 16

    print("\nRunning Positive Pass...")
    pos_vecs, pos_imgs = run_extraction(pipe, pos_prompts, args)
    pos_vecs["img_latent_hw"] = (h_latent, w_latent)
    pos_vecs["img_resolution"] = (args.height, args.width)
    torch.save(pos_vecs, os.path.join(args.save_dir, f_template.format("pos")))
    del pos_vecs
    torch.cuda.empty_cache()

    print("\nRunning Negative Pass...")
    neg_vecs, neg_imgs = run_extraction(pipe, neg_prompts, args)
    neg_vecs["img_latent_hw"] = (h_latent, w_latent)
    neg_vecs["img_resolution"] = (args.height, args.width)
    torch.save(neg_vecs, os.path.join(args.save_dir, f_template.format("neg")))
    del neg_vecs
    torch.cuda.empty_cache()

    if args.save_image_dir:
        os.makedirs(args.save_image_dir, exist_ok=True)
        save_grid(pos_imgs, os.path.join(args.save_image_dir, f"positive_{args.exp_type}_{args.num_prompts}_grid.png"))
        save_grid(neg_imgs, os.path.join(args.save_image_dir, f"negative_{args.exp_type}_{args.num_prompts}_grid.png"))

    print("Done.")


if __name__ == "__main__":
    main()
