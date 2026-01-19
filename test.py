import torch
import os
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Callable
from diffusers import StableDiffusion3Pipeline, FluxPipeline

# --- 1. Prompt Generation Logic ---

# --- 1. Prompt Generation (Restored) ---

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


# --- 2. Unified Extraction and Calculation ---

def calculate_steering_vector(pipe, prompts_pos, prompts_neg):
    print(prompts_pos, prompts_neg)
    all_pos_seq, all_pos_pooled = [], []
    all_neg_seq, all_neg_pooled = [], []
    
    is_flux = isinstance(pipe, FluxPipeline)
    
    for i in tqdm(range(len(prompts_pos)), desc="Encoding"):
        with torch.no_grad():
            # Unified Encoding Logic
            if is_flux:
                # Flux returns: (prompt_embeds, pooled_prompt_embeds, text_ids)
                out_pos = pipe.encode_prompt(prompt=[prompts_pos[i]], prompt_2=None, device=pipe.device, num_images_per_prompt=1)
                out_neg = pipe.encode_prompt(prompt=[prompts_neg[i]], prompt_2=None, device=pipe.device, num_images_per_prompt=1)
                p_seq, p_pool = out_pos[0], out_pos[1]
                n_seq, n_pool = out_neg[0], out_neg[1]
            else:
                # SD3 returns: (prompt_embeds, neg_prompt_embeds, pooled_pos, pooled_neg)
                out_pos = pipe.encode_prompt(prompt=[prompts_pos[i]], device=pipe.device, do_classifier_free_guidance=False)
                out_neg = pipe.encode_prompt(prompt=[prompts_neg[i]], device=pipe.device, do_classifier_free_guidance=False)
                p_seq, p_pool = out_pos[0], out_pos[2]
                n_seq, n_pool = out_neg[0], out_neg[2]

            all_pos_seq.append(p_seq.cpu())
            all_pos_pooled.append(p_pool.cpu())
            all_neg_seq.append(n_seq.cpu())
            all_neg_pooled.append(n_pool.cpu())
            
        # if i % 5 == 0:
        #     torch.cuda.empty_cache()

        # Concatenate all prompts
    pos_seq = torch.cat(all_pos_seq)
    neg_seq = torch.cat(all_neg_seq)
    pos_pool = torch.cat(all_pos_pooled)
    neg_pool = torch.cat(all_neg_pooled)

    print(pos_seq.shape, neg_seq.shape)
    print(pos_pool.shape, neg_pool.shape)

    text_embeddings_pos = {
        'sequence': pos_seq,
        'pooled': pos_pool
    }
    text_embeddings_neg = {
        'sequence': neg_seq,
        'pooled': neg_pool
    }

    return text_embeddings_pos, text_embeddings_neg

# --- 3. Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='concrete', choices=['style', 'concrete', 'switch'])
    parser.add_argument('--pos_concept', type=str, required=True)
    parser.add_argument('--neg_concept', type=str, default="")
    parser.add_argument('--num_prompts', type=int, default=20)
    parser.add_argument('--model_name', type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument('--save_dir', type=str, default="data_vectors/text_steering")
    parser.add_argument('--prompts_file', type=str, default="background.txt")
    args = parser.parse_args()

    # Load Model (Stripped down: No transformer/vae)
    is_flux = "flux" in args.model_name.lower()
    pipe_cls = FluxPipeline if is_flux else StableDiffusion3Pipeline
    dtype = torch.bfloat16 if is_flux else torch.float16
    
    print(f"Loading Text Encoders from {args.model_name}...")
    pipe = pipe_cls.from_pretrained(
        args.model_name, 
        torch_dtype=dtype,
        use_safetensors=True,
        transformer=None,
        vae=None,
        scheduler=None,
        device_map="balanced"
    )
    
    # Explicitly move ONLY the text encoders to the target device (GPU)
    # Map task name to prompt function
    PROMPT_FUNCS = {
        'style': get_prompts_style,
        'concrete': get_prompts_concrete,
        'switch': get_prompts_switch, # Added 'add' for completeness
    }
    p_func = PROMPT_FUNCS.get(args.task)
    if not p_func:
        raise ValueError(f"Invalid task: {args.task}")

    p_pos, p_neg = p_func(args.num_prompts, args.pos_concept, args.neg_concept, prompt_path=args.prompts_file)

    pos_embeds, neg_embeds = calculate_steering_vector(pipe, p_pos, p_neg)

    # Save
    os.makedirs(args.save_dir, exist_ok=True)

    # Save filenames
    embeds_pos_filename = f"{args.task}_{args.pos_concept}_{args.pos_concept}_prompts_{args.num_prompts}_pos_embeddings.pt"
    embeds_neg_filename = f"{args.task}_{args.neg_concept}_{args.pos_concept}_prompts_{args.num_prompts}_neg_embeddings.pt"
    # embeds_pos_filename = f"{args.task}_{args.neg_concept}_prompts_{args.num_prompts}_pos_embeddings.pt"
    # embeds_neg_filename = f"{args.task}_{args.neg_concept}_prompts_{args.num_prompts}_neg_embeddings.pt"

    torch.save(pos_embeds, os.path.join(args.save_dir, embeds_pos_filename))
    torch.save(neg_embeds, os.path.join(args.save_dir, embeds_neg_filename))

    print(f"\nSaved positive text embeddings to: {os.path.join(args.save_dir, embeds_pos_filename)}")
    print(f"Saved negative text embeddings to: {os.path.join(args.save_dir, embeds_neg_filename)}")
    # prefix = f"{args.task}_{args.pos_concept}_vs_{args.neg_concept or 'none'}"
    
    # torch.save(results["steering_vector"], os.path.join(args.save_dir, f"{prefix}_steering_vec.pt"))
    # # Optionally save full data for analysis
    # # torch.save(results["pos_metadata"], os.path.join(args.save_dir, f"{prefix}_pos_full.pt"))
    
    # print(f"\n✅ Steering Vector calculated and saved to {args.save_dir}")
    # print(f"Sequence Shape: {results['steering_vector']['sequence'].shape}")