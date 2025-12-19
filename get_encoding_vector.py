import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from tqdm import tqdm
import numpy as np
import os
import argparse
from typing import List, Tuple, Callable, Dict, Optional, Union

# Standardized type hint for prompt function return
PromptsTuple = Tuple[List[str], List[str]]

def get_imagenet_classes(num: int = 50) -> List[str]:
    """Reads a list of Imagenet classes from a file."""
    try:
        with open('imagenet_classes_animals.txt', 'r') as f:
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

def get_prompts_add(num=50, concept_pos='wering hat', concept_neg=None):
    # Adjust path as necessary for your environment
    try:
        with open("/home/jovyan/shares/SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs/background.txt", "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: background.txt not found. Using dummy prompts for 'add'.")
        lines = [f"dummy prompt {i}" for i in range(num)]
        
    prompts = lines[:num]
    prompts_pos = [f"{concept_pos} {p}" for p in prompts]
    if concept_neg:
        prompts_neg = [f"{concept_neg} {p}" for p in prompts]
    else:
        prompts_neg = prompts
    return prompts_pos, prompts_neg


def get_text_embeddings(
    pipe,
    prompt_func: Callable[..., PromptsTuple],
    num_prompts: int = 50,
    concept_pos: str = "Snoopy",
    concept_neg: Optional[str] = None,
    model_name: str = '',
) -> Tuple[Dict, Dict]:
    """
    Generates and extracts the sequence embeddings (prompt_embeds) 
    and pooled embeddings (pooled_prompt_embeds) for positive and negative prompts, 
    processing one prompt at a time to prevent CUDA OOM.
    """
    
    # 1. Get prompts
    prompts_pos, prompts_neg = prompt_func(num=num_prompts, concept_pos=concept_pos, concept_neg=concept_neg)
    print(f"Positive Prompts Sample: {prompts_pos[:3]}")
    print(f"Negative Prompts Sample: {prompts_neg[:3]}")
    
    # Ensure lists are of the same length
    min_len = min(len(prompts_pos), len(prompts_neg))
    prompts_pos = prompts_pos[:min_len]
    prompts_neg = prompts_neg[:min_len]

    # Containers for results
    all_prompt_embeds_pos, all_pooled_prompt_embeds_pos = [], []
    all_prompt_embeds_neg, all_pooled_prompt_embeds_neg = [], []
    
    # 2. Encode one prompt at a time
    for i in tqdm(range(min_len), desc="Encoding Prompts"):
        
        # Use a list of one prompt for the encoder
        p_pos = [prompts_pos[i]]
        p_neg = [prompts_neg[i]]
        
        with torch.no_grad():
            # Encode positive prompt (B=1)
            prompt_embeds_pos, pooled_prompt_embeds_pos, _ = pipe.encode_prompt(
                prompt=p_pos,
                prompt_2=None, 
                device=pipe.device,
                num_images_per_prompt=1,
                
            )
            
            # Encode negative prompt (B=1)
            prompt_embeds_neg, pooled_prompt_embeds_neg, _ = pipe.encode_prompt(
                prompt=p_neg,
                prompt_2=None,
                device=pipe.device,
                num_images_per_prompt=1,
               
            )

        # Store results (move to CPU immediately)
        all_prompt_embeds_pos.append(prompt_embeds_pos.cpu())
        all_pooled_prompt_embeds_pos.append(pooled_prompt_embeds_pos.cpu())
        all_prompt_embeds_neg.append(prompt_embeds_neg.cpu())
        all_pooled_prompt_embeds_neg.append(pooled_prompt_embeds_neg.cpu())
        
        # Explicitly clear cached memory after each prompt to be extra safe
        torch.cuda.empty_cache()


    # 3. Concatenate all single-prompt tensors into final batches
    final_prompt_embeds_pos = torch.cat(all_prompt_embeds_pos)
    final_pooled_prompt_embeds_pos = torch.cat(all_pooled_prompt_embeds_pos)
    final_prompt_embeds_neg = torch.cat(all_prompt_embeds_neg)
    final_pooled_prompt_embeds_neg = torch.cat(all_pooled_prompt_embeds_neg)


    # 4. Store and return the final embeddings dictionaries
    text_embeddings_pos = {
        'sequence': final_prompt_embeds_pos,
        'pooled': final_pooled_prompt_embeds_pos
    }
    text_embeddings_neg = {
        'sequence': final_prompt_embeds_neg,
        'pooled': final_pooled_prompt_embeds_neg
    }
    
    return text_embeddings_pos, text_embeddings_neg

# --- Main Execution ---

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Text embedding extraction parameters")
    parser.add_argument('--task', type=str, default='style', 
                        choices=['style', 'concrete', 'add'],
                        help='Prompt generation task type.')
    parser.add_argument('--exp_type', type=str, default='embeddings')
    parser.add_argument('--pos_concept', type=str, default='picasso', help='Concept for pos prompts.')
    parser.add_argument('--neg_concept', type=str, default='realistic', help='Concept for neg prompts.')
    parser.add_argument('--num_prompts', type=int, default=20, help='Number of prompts to use.')
    parser.add_argument('--save_dir', type=str, default='test_embeddings_flux', help='Directory to save results.')
    parser.add_argument('--model_name', type=str, 
                    default="stabilityai/stable-diffusion-3-medium",
                    help='Name of the Hugging Face model to load (e.g., SD3, Flux-Dev).')
    args = parser.parse_args()
    
    # --- Model Loading Logic (Optimized for Memory) ---
    model_name_lower = args.model_name.lower()
    
    # Determine the device (use CUDA if available)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'stable-diffusion-3' in model_name_lower or 'sd3' in model_name_lower:
        pipe_class = StableDiffusion3Pipeline
    elif 'flux' in model_name_lower:
        pipe_class = FluxPipeline
    else:
        raise ValueError(f"Unsupported model specified: {args.model_name}. Only SD3 and Flux models are supported.")
        
    print(f"Loading {args.model_name} with only text encoders on {DEVICE}...")

    # Load components with `device_map=None` to control memory usage manually
    # We only load components necessary for `encode_prompt`.
    pipe = pipe_class.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="balanced", # Disable automatic device mapping
        use_safetensors=True,
        # Set unnecessary components to None to avoid loading them
        transformer=None,
        vae=None,
        scheduler=None,
    )
    
    # Explicitly move ONLY the text encoders to the target device (GPU)
    # Map task name to prompt function
    PROMPT_FUNCS = {
        'style': get_prompts_style,
        'concrete': get_prompts_concrete,
        'add': get_prompts_add, # Added 'add' for completeness
    }
    
    prompt_func = PROMPT_FUNCS.get(args.task)
    if not prompt_func:
        raise ValueError(f"Invalid task: {args.task}")


    # Run extraction
    pos_embeds, neg_embeds = get_text_embeddings(
        pipe=pipe,
        prompt_func=prompt_func,
        num_prompts=args.num_prompts,
        concept_pos=args.pos_concept,
        concept_neg=args.neg_concept,
        model_name=args.model_name
    )
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save filenames
    embeds_pos_filename = f"{args.exp_type}_{args.neg_concept}_prompts_{args.num_prompts}_pos_embeddings.pt"
    embeds_neg_filename = f"{args.exp_type}_{args.neg_concept}_prompts_{args.num_prompts}_neg_embeddings.pt"

    torch.save(pos_embeds, os.path.join(args.save_dir, embeds_pos_filename))
    torch.save(neg_embeds, os.path.join(args.save_dir, embeds_neg_filename))

    print(f"\nSaved positive text embeddings to: {os.path.join(args.save_dir, embeds_pos_filename)}")
    print(f"Saved negative text embeddings to: {os.path.join(args.save_dir, embeds_neg_filename)}")