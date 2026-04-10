"""
FLUX DiT activation extraction — dual-stream version.

Collects BOTH image (4096×3072) and text (512×3072) branch activations
from double-stream blocks, with configurable extraction points.

Extraction points and what they capture:
  - "block"   : residual stream output (img, txt after full block including skip)
                 → what the next block actually sees. Best for steering application.
  - "attn"    : joint attention output (img_attn, txt_attn before gate + residual)
                 → where cross-stream interaction happens. Most concept-concentrated.
  - "norm1"   : post-LayerNorm, pre-attention (normalized → ~on unit sphere)
                 → geometrically clean for OT cost computation.
  - "norm2"   : post-LayerNorm, pre-MLP (normalized)
                 → same as norm1 but after attention residual.

Output format:
  {
    step_0: {
      "layer_0": {
        "img": tensor(n_samples, n_img_tokens, hidden_dim),    # 4096×3072
        "txt": tensor(n_samples, n_txt_tokens, hidden_dim),    # 512×3072
      },
      "layer_1": { ... },
      ...
    },
    step_1: { ... },
  }

This replaces the old format where only one branch was saved.
Compatible with compute_ot_steering_v3.py (just index into ["img"] or ["txt"]).
"""

import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from diffusers import FluxPipeline
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from PIL import Image


def get_imagenet_classes(num: int = 50, prompt_path='imagenet_classes_animals.txt') -> List[str]:
    try:
        with open(prompt_path, 'r') as f:
            return [line.strip() for i, line in enumerate(f) if i < num]
    except FileNotFoundError:
        return [f"class_{i}" for i in range(num)]

def get_prompts_style(num=50, concept_pos='anime', concept_neg=None, prompt_path='imagenet_classes_animals.txt'):
    classes = get_imagenet_classes(num, prompt_path=prompt_path)
    pos = [f"{cls}, {concept_pos} style" for cls in classes]
    neg = [f"{cls}, {concept_neg} style" if concept_neg else cls for cls in classes]
    return pos, neg

def get_prompts_concrete(num=50, concept_pos='Snoopy', concept_neg=None, prompt_path='imagenet_classes_animals.txt'):
    classes = get_imagenet_classes(num, prompt_path=prompt_path)
    pos = [f"{cls} with {concept_pos}" for cls in classes]
    neg = [f"{cls} with {concept_neg}" if concept_neg else cls for cls in classes]
    return pos, neg

def get_prompts_switch(num=50, concept_pos='Snoopy', concept_neg=None, prompt_path='imagenet_classes_animals.txt'):
    classes = get_imagenet_classes(num, prompt_path=prompt_path)
    pos = [f"{concept_pos} {cls}" for cls in classes]
    neg = [f"{concept_neg} {cls}" if concept_neg else cls for cls in classes]
    return pos, neg


def get_prompts_human_related(num=100, concept_pos='nudity', concept_neg=None, prompt_path=None):
    B = ['a girl', 'a boy', 'two men', 'two women', 'two people', 'a man', 'a woman', 'an old man', 'an old woman', 'boys', 'girls', 'man', 'woman', 'group of people', 'a human']
    
    C = ['', 'gloomy image', 'zoomed in', 'talking',  'on a beach', 'in a strange pose',  'realism', 
          'colorful background',  'smiling', ]
    
    prompts_pos = []
    prompts_neg = []
    for b in B:
        for c in C:
            prompts_pos.append(b+' '+c+', {}'.format(concept_pos))
            if concept_neg is not None:
                prompts_neg.append(b+' '+c+', {}'.format(concept_neg))
            else:
                prompts_neg.append(b+' '+c)

    
    return prompts_pos[:num], prompts_neg[:num]


PROMPT_FUNCS = {
    'style': get_prompts_style,
    'concrete': get_prompts_concrete,
    'switch': get_prompts_switch,
    'people': get_prompts_human_related,
}


# ---------------------------------------------------------------------------
# Dual-Stream Hook Manager
# ---------------------------------------------------------------------------

class DualStreamHookManager:
    """
    Manages forward hooks for FLUX double-stream blocks.
    
    Tracks diffusion steps by counting how many blocks have fired.
    When all hooked blocks fire once, that's one complete step.
    
    FLUX double-stream block output: (encoder_hidden_states, hidden_states)
                                      = (txt, img)
    
    For sub-modules:
      - attn output: (img_attn_out, txt_attn_out) — note: img first here
      - norm1 output: img_normalized (single tensor, not tuple)
      - norm1_context output: txt_normalized (single tensor, not tuple)
      - ff output: img_ff (single tensor)
      - ff_context output: txt_ff (single tensor)
    """

    def __init__(
        self,
        extraction_point: str,
        num_blocks: int,
        save_blocks: int,
        save_timesteps: int,
        use_cfg: bool = False,       # whether guidance_scale > 1 (CFG mode)
    ):
        self.extraction_point = extraction_point
        self.num_blocks = num_blocks
        self.save_blocks = save_blocks
        self.save_timesteps = save_timesteps
        self.use_cfg = use_cfg
        self.data: Dict = {}
        self.reset_state()

    def reset_state(self):
        self.current_step = 0
        self.hooks_fired_this_step = 0

    def _advance_step_counter(self):
        """Called after each hook fires. Increments step when all blocks done."""
        self.hooks_fired_this_step += 1

        # For norm-based extraction: we hook BOTH norm1 and norm1_context
        # per block, so total hooks per step = 2 * num_blocks
        hooks_per_step = self.num_blocks
        if self.extraction_point in ("norm1", "norm2"):
            hooks_per_step = self.num_blocks * 2  # img_norm + txt_norm per block

        if self.hooks_fired_this_step >= hooks_per_step:
            self.current_step += 1
            self.hooks_fired_this_step = 0

    def _should_save(self, block_idx: int) -> bool:
        return (
            block_idx < self.save_blocks
            and self.current_step < self.save_timesteps
        )

    def _store(self, step: int, block_idx: int, branch: str, tensor: torch.Tensor):
        """Store activation tensor for a specific step/block/branch."""
        if step not in self.data:
            self.data[step] = {}
        layer_key = f"layer_{block_idx}"
        if layer_key not in self.data[step]:
            self.data[step][layer_key] = {"img": [], "txt": []}
        self.data[step][layer_key][branch].append(tensor.detach().cpu())

    def _extract_conditional(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        If CFG is active, batch is [unconditional, conditional].
        We want the conditional half (second half).
        If no CFG (guidance_scale=1), take everything.
        """
        print(tensor.shape)
        if self.use_cfg and tensor.shape[0] > 1:
            return tensor[tensor.shape[0] // 2:]
        return tensor

    # --- Hook factories for each extraction point ---

    def make_block_hook(self, block_idx: int):
        """
        Hook on the entire FluxTransformerBlock.
        Output: (encoder_hidden_states, hidden_states) = (txt, img)
        """
        def hook_fn(module, input, output):
            if self._should_save(block_idx):
                txt, img = output[0], output[1]
                txt = self._extract_conditional(txt)
                img = self._extract_conditional(img)
                self._store(self.current_step, block_idx, "txt", txt)
                self._store(self.current_step, block_idx, "img", img)
            self._advance_step_counter()
        return hook_fn

    def make_attn_hook(self, block_idx: int):
        """
        Hook on the attention module.
        Output from FluxAttnProcessor: (img_attn_out, txt_attn_out)
        Note: some diffusers versions return just hidden_states if no encoder_hidden_states.
        """
        def hook_fn(module, input, output):
            if self._should_save(block_idx):
                if isinstance(output, tuple) and len(output) >= 2:
                    img_attn, txt_attn = output[0], output[1]
                else:
                    # Single stream or unexpected format — save as img
                    img_attn = output if not isinstance(output, tuple) else output[0]
                    txt_attn = None
                print(img_attn.shape, txt_attn.shape, block_idx, self.current_step)
                img_attn = self._extract_conditional(img_attn)
                self._store(self.current_step, block_idx, "img", img_attn)
                if txt_attn is not None:
                    txt_attn = self._extract_conditional(txt_attn)
                    self._store(self.current_step, block_idx, "txt", txt_attn)

            self._advance_step_counter()
        return hook_fn

    def make_norm_hook(self, block_idx: int, branch: str):
        """
        Hook on norm1 (img) or norm1_context (txt).
        Output: single tensor (the normalized hidden states).
        
        For norm-based extraction, we register TWO hooks per block
        (one for img norm, one for txt norm), so step counting is adjusted.
        """
        def hook_fn(module, input, output):
            if self._should_save(block_idx):
                act = self._extract_conditional(output)
                self._store(self.current_step, block_idx, branch, act)
            self._advance_step_counter()
        return hook_fn

    def make_ff_hook(self, block_idx: int, branch: str):
        """
        Hook on ff (img MLP) or ff_context (txt MLP).
        Output: single tensor.
        """
        def hook_fn(module, input, output):
            if self._should_save(block_idx):
                act = self._extract_conditional(output)
                self._store(self.current_step, block_idx, branch, act)
            self._advance_step_counter()
        return hook_fn

    def aggregate(self) -> Dict:
        """Stack per-sample lists into tensors."""
        result = {}
        for step, layers in self.data.items():
            result[step] = {}
            for layer_key, branches in layers.items():
                result[step][layer_key] = {}
                for branch, tensor_list in branches.items():
                    if tensor_list:
                        result[step][layer_key][branch] = torch.stack(tensor_list)
        return result


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def register_hooks(
    pipe,
    manager: DualStreamHookManager,
    extraction_point: str,
    num_blocks: int,
) -> List:
    """
    Register forward hooks on the appropriate sub-modules.
    Returns list of hook handles for cleanup.
    """
    transformer = pipe.transformer
    handles = []

    # FLUX double-stream blocks
    blocks = transformer.transformer_blocks

    for idx in range(min(num_blocks, len(blocks))):
        
        block = blocks[idx]
        print(block)
       
        if extraction_point == "block":
            # Hook the entire block — output is (txt, img)
            h = block.register_forward_hook(manager.make_block_hook(idx))
            handles.append(h)

        elif extraction_point == "attn":
            # Hook the attention module — output is (img_attn, txt_attn)
            h = block.attn.register_forward_hook(manager.make_attn_hook(idx))
            handles.append(h)

        elif extraction_point == "norm1":
            # Hook both norm modules — pre-attention normalized activations
            h_img = block.norm1.register_forward_hook(
                manager.make_norm_hook(idx, "img")
            )
            h_txt = block.norm1_context.register_forward_hook(
                manager.make_norm_hook(idx, "txt")
            )
            handles.extend([h_img, h_txt])

        elif extraction_point == "norm2":
            # Hook both norm2 modules — pre-MLP normalized activations
            h_img = block.norm2.register_forward_hook(
                manager.make_norm_hook(idx, "img")
            )
            h_txt = block.norm2_context.register_forward_hook(
                manager.make_norm_hook(idx, "txt")
            )
            handles.extend([h_img, h_txt])

        elif extraction_point == "ff":
            # Hook MLP outputs for both branches
            h_img = block.ff.register_forward_hook(
                manager.make_ff_hook(idx, "img")
            )
            h_txt = block.ff_context.register_forward_hook(
                manager.make_ff_hook(idx, "txt")
            )
            handles.extend([h_img, h_txt])

        else:
            raise ValueError(f"Unknown extraction_point: {extraction_point}")

    return handles


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_extraction(pipe, prompts, args) -> Tuple[Dict, List]:
    """
    Run inference and collect activations from both streams.
    
    Returns:
        vectors: {step: {layer_N: {"img": tensor, "txt": tensor}}}
        images: list of PIL images
    """
    use_cfg = args.gs > 1.0

    manager = DualStreamHookManager(
        extraction_point=args.extraction_point,
        num_blocks=args.num_layers,
        save_blocks=args.num_layers,
        save_timesteps=args.save_timesteps,
        use_cfg=use_cfg,
    )

    handles = register_hooks(pipe, manager, args.extraction_point, args.num_layers)
    all_images = []

    try:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Extracting"):
            batch = prompts[i:i + args.batch_size]
            manager.reset_state()

            generators = [
                torch.Generator("cuda").manual_seed(42000 + i * 10 + j)
                for j in range(len(batch))
            ]

            res = pipe(
                batch,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.gs,
                height=args.height,
                width=args.width,
                generator=generators,
            )
            all_images.extend(res.images)
    finally:
        for h in handles:
            h.remove()

    vectors = manager.aggregate()

    # Print shape summary
    for step in sorted(vectors.keys()):
        for layer_key in sorted(vectors[step].keys()):
            branches = vectors[step][layer_key]
            shapes = {b: tuple(t.shape) for b, t in branches.items()}
            print(f"  step {step} {layer_key}: {shapes}")
        break  # only print first step

    return vectors, all_images


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FLUX dual-stream activation extraction"
    )

    # Task / concept
    parser.add_argument('--task', type=str, default='style',
                        choices=['style', 'concrete', 'switch', 'people'])
    parser.add_argument('--exp_type', type=str, default='3d')
    parser.add_argument('--pos_concept', type=str, default='picasso')
    parser.add_argument('--neg_concept', type=str, default='realistic')
    parser.add_argument('--num_prompts', type=int, default=10)
    parser.add_argument('--prompt_path', type=str, default='imagenet_classes.txt')

    # Model
    parser.add_argument('--model_name', type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument('--gs', type=float, default=4.5)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--height', type=int, default=1024,
                        help="Generation height. Latent grid = height//16")
    parser.add_argument('--width', type=int, default=1024,
                        help="Generation width. Latent grid = width//16")

    # Extraction config
    parser.add_argument('--extraction_point', type=str, default='block',
                        choices=['block', 'attn', 'norm1', 'norm2', 'ff'],
                        help="Where to hook: "
                             "'block' = residual stream (recommended for steering), "
                             "'attn' = post joint-attention, "
                             "'norm1' = post-LayerNorm pre-attention (on sphere), "
                             "'norm2' = post-LayerNorm pre-MLP, "
                             "'ff' = post-MLP")
    parser.add_argument('--num_layers', type=int, default=19,
                        help="Number of double-stream blocks to hook (FLUX.1-dev has 19)")
    parser.add_argument('--save_timesteps', type=int, default=4,
                        help="Save activations for the first N diffusion steps")

    # Output
    parser.add_argument('--save_dir', type=str, default='test_vectors_dci/style')
    parser.add_argument('--save_image_dir', type=str, default=None)

    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading {args.model_name}...")
    pipe = FluxPipeline.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="balanced"
    )

    # --- Generate prompts ---
    func = PROMPT_FUNCS.get(args.task)
    if not func:
        raise ValueError(f"Invalid task: {args.task}")

    pos_prompts, neg_prompts = func(
        num=args.num_prompts,
        concept_pos=args.pos_concept,
        concept_neg=args.neg_concept,
        prompt_path=args.prompt_path,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Prompts: {len(pos_prompts)} pos, {len(neg_prompts)} neg")
    print(f"Extraction: {args.extraction_point}, blocks: {args.num_layers}, "
          f"timesteps: {args.save_timesteps}")

    # --- File naming ---
    f_template = (
        f"{args.exp_type}_{args.neg_concept}_gs_{args.gs}_"
        f"prompts_{args.num_prompts}_{{}}_{args.extraction_point}.pt"
    )

    # Latent dimensions metadata (needed for resolution adaptation at inference)
    h_latent = args.height // 16
    w_latent = args.width // 16
    print(f"Latent grid: {h_latent}×{w_latent} = {h_latent * w_latent} image tokens")

    # --- Positive pass ---
    print("\nRunning Positive Pass...")
    pos_vecs, pos_imgs = run_extraction(pipe, pos_prompts, args)
    pos_vecs["img_latent_hw"] = (h_latent, w_latent)
    pos_vecs["img_resolution"] = (args.height, args.width)
    torch.save(pos_vecs, os.path.join(args.save_dir, f_template.format("pos")))
    del pos_vecs
    torch.cuda.empty_cache()

    # --- Negative pass ---
    print("\nRunning Negative Pass...")
    neg_vecs, neg_imgs = run_extraction(pipe, neg_prompts, args)
    neg_vecs["img_latent_hw"] = (h_latent, w_latent)
    neg_vecs["img_resolution"] = (args.height, args.width)
    torch.save(neg_vecs, os.path.join(args.save_dir, f_template.format("neg")))
    del neg_vecs
    torch.cuda.empty_cache()

    # --- Save image grids ---
    if args.save_image_dir and pos_imgs and neg_imgs:
        os.makedirs(args.save_image_dir, exist_ok=True)
        to_tensor = ToTensor()

        def save_grid(images, filename):
            if not images:
                return
            nrow = int(len(images) ** 0.5)
            tensors = [to_tensor(img) for img in images]
            grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
            img = Image.fromarray(
                (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            img.save(os.path.join(args.save_image_dir, filename))

        save_grid(pos_imgs, f"positive_{args.exp_type}_{args.num_prompts}_grid.png")
        save_grid(neg_imgs, f"negative_{args.exp_type}_{args.num_prompts}_grid.png")
        print(f"Saved image grids to {args.save_image_dir}")

    print("Done.")