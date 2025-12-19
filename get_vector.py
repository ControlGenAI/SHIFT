import torch
import os
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Callable
from contextlib import contextmanager
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from PIL import Image


# Standardized type hint for prompt function return
PromptsTuple = Tuple[List[str], List[str]]

def get_imagenet_classes(num: int = 50, prompt_path='imagenet_classes_animals.txt') -> List[str]:
    """Reads a list of Imagenet classes from a file."""
    try:
        with open(prompt_path, 'r') as f:
            return [line.strip() for i, line in enumerate(f) if i < num]
    except FileNotFoundError:
        print("Warning: 'imagenet_classes.txt' not found. Using a dummy list.")
        return [f"class_{i}" for i in range(num)]

# --- Prompt Generation Functions ---

def get_prompts_style(num=50, concept_pos='anime', concept_neg=None):
    classes = get_imagenet_classes(num)
    return (
        [f"{cls}, {concept_pos} style" for cls in classes],
        [f"{cls}, {concept_neg} style" if concept_neg else cls for cls in classes]
    )


def get_prompts_concrete(num=50, concept_pos='Snoopy', concept_neg=None):
    classes = get_imagenet_classes(num)
    return (
        [f"{cls} with {concept_pos}" for cls in classes],
        [f"{cls} with {concept_neg}" if concept_neg else cls for cls in classes]
    )


def get_attention_steering_vector_multistep(
    pipe,
    prompt_func: Callable[..., PromptsTuple],
    activations: str = 'attn_enc', 
    num_prompts: int = 50,
    concept_pos: str = "Snoopy",
    concept_neg: Optional[str] = None,
    batch_size: int = 8,
    layer_indices: Optional[List[int]] = None,
    num_inference_steps: int = 20,
    gs: float = 4.5,
    seed = 42,
    save_image_dir: Optional[str] = None,
    exp_type='3d',
    model_name='',
) -> Tuple[Dict, Dict]:
    """
    Extracts activations from the UNet's attention or feed-forward layers 
    for positive and negative prompt sets across multiple diffusion steps.
    """
    
    step_activations: Dict[int, Dict[int, List[List[torch.Tensor]]]] = {}
    
    images_pos: List[Image.Image] = []
    images_neg: List[Image.Image] = []

    # Get prompts
    prompts_pos, prompts_neg = prompt_func(num=num_prompts, concept_pos=concept_pos, concept_neg=concept_neg)
    print(f"Positive Prompts Sample: {prompts_pos[:]}")
    print(f"Negative Prompts Sample: {prompts_neg[:]}")

    act_idx = 1 if gs != 1 else 0
    
    if 'flux' in model_name.lower():
        act_extraction = lambda output: output[1][0].detach().cpu() if len(output) == 2 else output[:, :512, :].detach().cpu()
        end_module_name = 'attn'
    else:
        if activations == 'attn_enc' or activations == 'attn_im':
            if activations == 'attn_enc':
                act_extraction = lambda output: output[1][act_idx].detach().cpu()
            else:
                act_extraction = lambda output: output[0][act_idx].detach().cpu()
            end_module_name = 'attn'
        elif activations == 'ff':
            act_extraction = lambda output: output[act_idx].detach().cpu()
            end_module_name = 'ff_context'
        else:
            raise ValueError(f"Unknown activation type: {activations}")

    current_step = 0 # Nonlocal step counter
    
    def attn_hook(layer_idx: int, is_positive_prompt: int):
        """Creates a closure for the forward hook."""
        def hook(module, input, output):
            nonlocal current_step
            try:
                print(output[:, :, :].shape)
            except:
                print(output[0].shape, output[1].shape)
            
            act = act_extraction(output) 
            print(layer_idx, current_step)
            if  len(output) == 2 :
                print('save this part')
                if current_step not in step_activations:
                    step_activations[current_step] = {}
                if layer_idx not in step_activations[current_step]:
                    step_activations[current_step][layer_idx] = [[], []]  # [pos_acts, neg_acts]

                # Append activation to the correct list (pos or neg)
                step_activations[current_step][layer_idx][is_positive_prompt].append(act)

            total_hooked_layers = len(hook_handles)

            if layer_idx == (total_hooked_layers - 1): # If this is the last registered layer
                current_step += 1    
        return hook

    hook_handles = []
    idx_layers = 0
    transformer_modules = list(pipe.transformer.named_modules())
    
    # First, count how many layers we'll actually hook for correct total_hooked_layers calculation
    modules_to_hook = []
    for idx, (name, module) in enumerate(transformer_modules):
        if name.endswith(end_module_name) and (layer_indices is None or idx in layer_indices):
            modules_to_hook.append((idx, name, module))
            
    # Now register hooks with a layer index relative to the *hooked* layers
    for relative_idx, (abs_idx, name, module) in enumerate(modules_to_hook):
        hook_handles.append(module.register_forward_hook(attn_hook(relative_idx, is_positive_prompt=0)))
    
    # Store the final total count
    total_hooked_layers = len(hook_handles)

    # Generation loop
    all_step_vectors_pos: Dict[int, Dict[str, torch.Tensor]] = {}
    all_step_vectors_neg: Dict[int, Dict[str, torch.Tensor]] = {}
    
    # Ensure prompts lists are of the same length
    min_len = min(len(prompts_pos), len(prompts_neg))
    prompts_pos = prompts_pos[:min_len]
    prompts_neg = prompts_neg[:min_len]

    for i in tqdm(range(0, len(prompts_pos), batch_size), desc="Processing Batches"):
        batch_pos = prompts_pos[i:i+batch_size]
        batch_neg = prompts_neg[i:i+batch_size]

        seeds: List[int] = []
        
        # Re-register hooks for positive pass to set the `is_positive_prompt` flag in the closure
        for h in hook_handles: h.remove() # Clean up existing
        hook_handles = []
        for relative_idx, (abs_idx, name, module) in enumerate(modules_to_hook):
            hook_handles.append(module.register_forward_hook(attn_hook(relative_idx, is_positive_prompt=0)))
            
        current_step = 0 # Reset step counter

        for p_idx, p in enumerate(batch_pos):
            seed = 42000 + i*10 + p_idx
            #seed = 42
            seeds.append(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            print(p, seed)
            #p = "Two kids are playing baseball in Wii Sports cyberpunk style"
            with torch.no_grad():
                # Call pipe and let hooks record activations
                res = pipe(p, num_inference_steps=num_inference_steps, guidance_scale=gs, generator=generator)
                if save_image_dir:
                    images_pos.append(res.images[0])
            current_step = 0 # Reset for the next generation

        # 2. NEGATIVE PASS
        for h in hook_handles: h.remove() # Clean up existing
        hook_handles = []
        for relative_idx, (abs_idx, name, module) in enumerate(modules_to_hook):
            hook_handles.append(module.register_forward_hook(attn_hook(relative_idx, is_positive_prompt=1)))

        current_step = 0 # Reset step counter
        for p_idx, p in enumerate(batch_neg):
            seed = seeds[p_idx] # Use same seed as positive
            generator = torch.Generator(device="cuda").manual_seed(seed)
            print(p, seed)
            with torch.no_grad():
                res = pipe(p, num_inference_steps=num_inference_steps, guidance_scale=gs, generator=generator)
                if save_image_dir:
                    images_neg.append(res.images[0])
            current_step = 0 # Reset for the next generation


    # 3. COMPUTE AND STORE ACTIVATION TENSORS
    # Clean up hooks
    for h in hook_handles:
        h.remove()

    for step in tqdm(step_activations, desc="Aggregating Activations"):
        all_step_vectors_pos[step] = {}
        all_step_vectors_neg[step] = {}
        for layer in step_activations[step]:
            pos_acts, neg_acts = step_activations[step][layer]
                
            # Stack tensors: (batch_size, sequence_length, feature_dim)
            if pos_acts:
                pos = torch.stack(pos_acts)
                all_step_vectors_pos[step][f"layer_{layer}"] = pos.cpu()
            if neg_acts:
                neg = torch.stack(neg_acts)
                all_step_vectors_neg[step][f"layer_{layer}"] = neg.cpu()

    # 4. SAVE IMAGES
    if save_image_dir and images_pos and images_neg:
        os.makedirs(save_image_dir, exist_ok=True)
        to_tensor = ToTensor()
        
        # Determine the best grid layout (closest to square)
        def save_grid_image(images: List[Image.Image], filename: str):
            if not images: return
            n = len(images)
            nrow = int(n**0.5) # Square root for grid size
            tensors = [to_tensor(img) for img in images]
            grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
            # Convert grid tensor back to PIL Image
            img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img.save(os.path.join(save_image_dir, filename))

        save_grid_image(
            images_pos, 
            f"positive_{exp_type}_{num_prompts}_grid.png"
        )
        save_grid_image(
            images_neg, 
            f"negative_{'baseline'}_vs_{exp_type}_{num_prompts}_grid.png"
        )

        print(f"Saved image grids to {save_image_dir}")

    return all_step_vectors_pos, all_step_vectors_neg

# --- Main Execution ---

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Steering vector extraction parameters")
    parser.add_argument('--task', type=str, default='style', 
                        choices=['style', 'positions', 'background', 'concrete', 'objects', 'add', 'human_related', 'human_emotions'],
                        help='Prompt generation task type.')
    parser.add_argument('--exp_type', type=str, default='3d')
    parser.add_argument('--pos_concept', type=str, default='picasso', help='Concept for pos prompts (e.g., picasso, dog, nudity).')
    parser.add_argument('--neg_concept', type=str, default='realistic', help='Concept for neg prompts (e.g., realistic, cat, SFW).')
    parser.add_argument('--num_prompts', type=int, default=10, help='Number of prompts to use.')
    # Seed is no longer used for generation, only for numpy/random if needed globally
    parser.add_argument('--gs', type=float, default=4.5, help='Guidance scale.')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps.')
    parser.add_argument('--save_dir', type=str, default='test_vectors_dci/style', help='Directory to save results.')
    parser.add_argument('--activations_type', type=str, default='attn_enc', choices=['attn_enc', 'attn_im', 'ff'], help='Type of activations to save.')
    parser.add_argument('--save_image_dir', type=str, default=None, help='Directory to save image grids. Set to None to skip saving images.')
    parser.add_argument('--model_name', type=str, 
                    default="stabilityai/stable-diffusion-3.5-medium",
                    help='Name of the Hugging Face model to load (e.g., SD3, Flux-Dev).')
    args = parser.parse_args()

    model_name = args.model_name
    model_name_lower = model_name.lower()
    pipe = None # Initialize pipe

    if 'stable-diffusion-3' in model_name_lower or 'sd3' in model_name_lower:
        # 1. SD3 Models
        from diffusers import StableDiffusion3Pipeline
        print(f"Detected Stable Diffusion 3 model. Loading with StableDiffusion3Pipeline...")
        pipe_class = StableDiffusion3Pipeline
        
    elif 'flux' in model_name_lower:
        from diffusers import FluxPipeline
        print(f"Detected Flux model. Loading with FluxPipeline...")
        pipe_class = FluxPipeline
        
    else:
        # 3. Handle Unsupported Models
        raise ValueError(f"Unsupported model specified: {model_name}. Only SD3 and Flux models are supported.")
        
    
    # Load the selected pipeline class
    pipe = pipe_class.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="balanced",
        use_safetensors=True
    )
    print(f"Model {model_name} loaded successfully.")

    # Map task name to prompt function
    PROMPT_FUNCS = {
        'style': get_prompts_style,
        'concrete': get_prompts_concrete,
    }
    
    prompt_func = PROMPT_FUNCS.get(args.task)
    if not prompt_func:
        raise ValueError(f"Invalid task: {args.task}")


    # Run extraction
    pos_vectors, neg_vectors = get_attention_steering_vector_multistep(
        pipe=pipe,
        activations=args.activations_type,
        prompt_func=prompt_func,
        num_prompts=args.num_prompts,
        concept_pos=args.pos_concept,
        concept_neg=args.neg_concept,
        batch_size=1, # Batch size 1 for simplicity and seed control, as in original code
        num_inference_steps=args.num_inference_steps,
        gs=args.gs,
        save_image_dir=args.save_image_dir,
        exp_type=args.exp_type,
        model_name=args.model_name
    )
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create cleaner filenames
    pos_filename = f"{args.exp_type}_{args.neg_concept}_gs_{args.gs}_prompts_{args.num_prompts}_pos_{args.activations_type}.pt"
    neg_filename = f"{args.exp_type}_{args.neg_concept}_gs_{args.gs}_prompts_{args.num_prompts}_neg_{args.activations_type}.pt"

    torch.save(pos_vectors, os.path.join(args.save_dir, pos_filename))
    torch.save(neg_vectors, os.path.join(args.save_dir, neg_filename))
    
    print(f"\nSaved positive vectors to: {os.path.join(args.save_dir, pos_filename)}")
    print(f"Saved negative vectors to: {os.path.join(args.save_dir, neg_filename)}")