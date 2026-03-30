import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from diffusers import StableDiffusion3Pipeline, FluxPipeline
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


class SteeringHookManager:
    """Manages forward hooks and accurately tracks diffusion steps."""
    def __init__(self, handler_fn, activations_type, act_idx, num_layers, save_layers=None):
        self.handler_fn = handler_fn
        self.activations_type = activations_type
        self.act_idx = act_idx
        self.num_layers = num_layers
        self.save_layers = save_layers
        self.data = {} 
        self.reset_state()

    def reset_state(self):
        self.current_step = 0
        self.layers_called_in_step = 0

    def hook_fn(self, layer_idx: int):
        def wrapper(module, input, output):
            # Extract activation based on model type (SD3 vs Flux)
            act = self.handler_fn(output, self.activations_type, self.act_idx)
            if layer_idx < self.save_layers and self.current_step < 1:
                if self.current_step not in self.data:
                    self.data[self.current_step] = {}
                if layer_idx not in self.data[self.current_step]:
                    self.data[self.current_step][layer_idx] = []

                print(self.current_step, layer_idx, act.shape)
                self.data[self.current_step][layer_idx].append(act.detach().cpu())

            self.layers_called_in_step += 1
            if self.layers_called_in_step >= self.num_layers:
                self.current_step += 1
                self.layers_called_in_step = 0
        return wrapper

# Mapping for Model-Specific Logic
MODEL_HANDLERS = {
    'sd3': lambda out, t, idx: out[1][idx] if t == 'attn_enc' else (out[0][idx] if t == 'attn_im' else out[idx]),
    'flux': lambda out, t, idx: (out[1] if isinstance(out, tuple) else out)[:, :, :]
}


def run_extraction(pipe, model_type, prompts, args):
    # Setup configuration
    handler = MODEL_HANDLERS[model_type]
    layer_suffix = 'attn' if 'attn' in args.activations_type else 'ff_context'
    if model_type == 'flux' and 'ff' in args.activations_type: layer_suffix = 'ff'
    
    act_idx = 1 if args.gs > 1.0 else 0
    
    modules_to_hook = [
        (name, mod) for idx, (name, mod) in enumerate(pipe.transformer.named_modules())
        if name.endswith(layer_suffix)
    ]

    manager = SteeringHookManager(handler, args.activations_type, act_idx, len(modules_to_hook), save_layers=args.num_layers)
    all_images = []

    handles = []
    for i, (name, mod) in enumerate(modules_to_hook):
        handles.append(mod.register_forward_hook(manager.hook_fn(i)))

    try:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Extracting {args.task}"):
            batch = prompts[i:i+args.batch_size]
            manager.reset_state()
            generators = [torch.Generator("cuda").manual_seed(42000 + i*10 + j) for j in range(len(batch))]
            
            res = pipe(batch, num_inference_steps=args.num_inference_steps, guidance_scale=args.gs, generator=generators)
            all_images.extend(res.images)
    finally:
        for h in handles: h.remove()

    # Aggregate results
    vectors = {step: {f"layer_{l}": torch.stack(t) for l, t in layers.items()} for step, layers in manager.data.items()}
    return vectors, all_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='style', choices=['style', 'concrete', 'switch', 'people'])
    parser.add_argument('--exp_type', type=str, default='3d')
    parser.add_argument('--pos_concept', type=str, default='picasso')
    parser.add_argument('--neg_concept', type=str, default='realistic')
    parser.add_argument('--num_prompts', type=int, default=10)
    parser.add_argument('--gs', type=float, default=4.5)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='test_vectors_dci/style')
    parser.add_argument('--activations_type', type=str, default='attn_enc', choices=['attn_enc', 'attn_im', 'ff'])
    parser.add_argument('--save_image_dir', type=str, default=None)
    parser.add_argument('--prompt_path', type=str, default='imagenet_classes.txt')
    parser.add_argument('--model_name', type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument('--num_layers', type=int, default=19)
    
    args = parser.parse_args()

    # Model Selection
    m_lower = args.model_name.lower()
    m_type = 'flux' if 'flux' in m_lower else 'sd3'
    pipe_cls = FluxPipeline if m_type == 'flux' else StableDiffusion3Pipeline

    print(f"Loading {args.model_name}...")
    pipe = pipe_cls.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="balanced")

    # Prompt Selection
    PROMPT_FUNCS = {
        'style': get_prompts_style,
        'concrete': get_prompts_concrete,
        'switch': get_prompts_switch, 
        'people': get_prompts_human_related,
    }
    func = PROMPT_FUNCS.get(args.task)
    if not func:
        raise ValueError(f"Invalid task: {args.task}")

    pos_p, neg_p = func(num=args.num_prompts, concept_pos=args.pos_concept, concept_neg=args.neg_concept, prompt_path=args.prompt_path)

    os.makedirs(args.save_dir, exist_ok=True)

    print("Prompts:")
    print(pos_p, neg_p)
    print("Running Positive Pass...")
    pos_vecs, pos_imgs = run_extraction(pipe, m_type, pos_p, args)
    f_template = f"{args.exp_type}_{args.neg_concept}_gs_{args.gs}_prompts_{args.num_prompts}_{{}}_{args.activations_type}_1.pt"
   
    torch.save(pos_vecs, os.path.join(args.save_dir, f_template.format("pos")))
    del pos_vecs 
    print("Running Negative Pass...")
    neg_vecs, neg_imgs = run_extraction(pipe, m_type, neg_p, args)

    # Save Results
    os.makedirs(args.save_dir, exist_ok=True)
    f_template = f"{args.exp_type}_{args.neg_concept}_gs_{args.gs}_prompts_{args.num_prompts}_{{}}_{args.activations_type}_1.pt"
   
    torch.save(neg_vecs, os.path.join(args.save_dir, f_template.format("neg")))
    del neg_vecs

    if args.save_image_dir and pos_imgs and neg_imgs:
        os.makedirs(args.save_image_dir, exist_ok=True)
        to_tensor = ToTensor()
        
        def save_grid_image(images: List[Image.Image], filename: str):
            if not images: return
            n = len(images)
            nrow = int(n**0.5) 
            tensors = [to_tensor(img) for img in images]
            grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
            img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img.save(os.path.join(args.save_image_dir, filename))

        save_grid_image(
            pos_imgs, 
            f"positive_{args.exp_type}_{args.num_prompts}_grid.png"
        )
        save_grid_image(
            neg_imgs, 
            f"negative_{'baseline'}_vs_{args.exp_type}_{args.num_prompts}_grid.png"
        )


        print(f"Saved image grids to {args.save_image_dir}")

    print("Success! Vectors saved.")