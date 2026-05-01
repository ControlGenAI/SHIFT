import torch
import os
import argparse
from tqdm import tqdm
from typing import List
from diffusers import StableDiffusion3Pipeline, FluxPipeline


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



def calculate_steering_vector(pipe, prompts_pos, prompts_neg):
    if len(prompts_pos) != len(prompts_neg):
        raise ValueError(
            f"Prompt count mismatch: {len(prompts_pos)} positive vs {len(prompts_neg)} negative prompts."
        )
    print("Prompts:")
    #print(prompts_pos, prompts_neg)
    all_pos_seq, all_pos_pooled = [], []
    all_neg_seq, all_neg_pooled = [], []
    
    is_flux = isinstance(pipe, FluxPipeline)

    print(prompts_pos[:15], prompts_neg[:15])
    
    for i in tqdm(range(len(prompts_pos)), desc="Encoding"):
        with torch.no_grad():
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
    parser.add_argument('--task', type=str, default='concrete', choices=['style', 'concrete', 'switch', 'people'])
    parser.add_argument('--pos_concept', type=str, required=True)
    parser.add_argument('--neg_concept', type=str, default="")
    parser.add_argument('--num_prompts', type=int, default=20)
    parser.add_argument('--model_name', type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument('--save_dir', type=str, default="data_vectors/text_steering")
    parser.add_argument('--prompt_path', type=str, default='imagenet_classes.txt')
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

    PROMPT_FUNCS = {
        'style': get_prompts_style,
        'concrete': get_prompts_concrete,
        'switch': get_prompts_switch, # Added 'add' for completeness
        'people': get_prompts_human_related,
    }
    p_func = PROMPT_FUNCS.get(args.task)
    if not p_func:
        raise ValueError(f"Invalid task: {args.task}")

    p_pos, p_neg = p_func(args.num_prompts, args.pos_concept, args.neg_concept, prompt_path=args.prompt_path)

    pos_embeds, neg_embeds = calculate_steering_vector(pipe, p_pos, p_neg)

    # Save
    os.makedirs(args.save_dir, exist_ok=True)

    # Save filenames
    embeds_pos_filename = f"{args.task}_{args.pos_concept}_{args.pos_concept}_prompts_{args.num_prompts}_pos_embeddings.pt"
    embeds_neg_filename = f"{args.task}_{args.neg_concept}_{args.pos_concept}_prompts_{args.num_prompts}_neg_embeddings.pt"
    torch.save(pos_embeds, os.path.join(args.save_dir, embeds_pos_filename))
    torch.save(neg_embeds, os.path.join(args.save_dir, embeds_neg_filename))

    print(f"\nSaved positive text embeddings to: {os.path.join(args.save_dir, embeds_pos_filename)}")
    print(f"Saved negative text embeddings to: {os.path.join(args.save_dir, embeds_neg_filename)}")
