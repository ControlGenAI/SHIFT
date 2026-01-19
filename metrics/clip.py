import torch
from PIL import Image
from tqdm import tqdm
import os
import random
import numpy as np
import argparse
from transformers import CLIPProcessor, CLIPModel
import natsort
import json


# --- Configuration & Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILENAME = 'metrics_clip_reward.pt'

clip_model = None
clip_processor = None
try:
    print("Initializing CLIP Model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
except Exception as e:
    print(f"Error loading CLIP: {e}. CLIP metrics will be skipped.")

# --- CLIP Metric Functions ---

def get_clip_image_embedding(img: Image.Image) -> torch.Tensor:
    """Gets the normalized CLIP image feature vector."""
    if clip_model is None: return None
    inputs = clip_processor(images=img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(pixel_values=pixel_values)
        return image_embeds[0] / image_embeds[0].norm()

def compute_clip_image_score(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
    """Computes CLIP Image-to-Image Cosine Similarity from normalized features."""
    if feat1 is None or feat2 is None: return None
    return torch.dot(feat1, feat2).item()

def compute_clip_score(image: Image.Image, text_prompt: str) -> float:
    """Computes CLIP Score (Image-to-Text Cosine Similarity)."""
    if clip_model is None: return None
    inputs = clip_processor(text=text_prompt, images=image, return_tensors="pt", padding=True)
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        outputs = clip_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        # Normalized embeddings
        image_embed = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embed = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        
        return (image_embed * text_embed).sum(dim=1).item()

# def compute_image_reward_mock(image: Image.Image, text_prompt: str) -> float:
#     """Mock Image Reward Score (Replace with actual model if needed)."""
#     return random.uniform(0.5, 1.0) 

# --- Main Logic ---

def compute_clip_reward_metrics(origin_dir, target_dirs, image_prompts, style_prompt, save_dir):
    """
    Computes CLIP Image Similarity, CLIP Score (vs. style prompt), and Image Reward.
    """
    if not clip_model:
        print("CLIP model not available. Skipping metric computation.")
        return

    metrics_clip_reward = {}

    prompts = []
    
    with open(image_prompts[0], 'r', encoding='utf-8') as f:
        for line in f:
            # Use .strip() to remove leading/trailing whitespace, 
            # including the newline character '\n'
            prompt = line.strip()
            
            # Only add non-empty lines to the list
            if prompt:
                prompts.append(prompt)

    # Use os.listdir and sorted to get a list of image files in a consistent order
    try:
        all_origin_files = {f: f.rsplit('.', 1)[0] for f in os.listdir(origin_dir) if os.path.isfile(os.path.join(origin_dir, f))}
        
        # Filter files to only those that match the base prompts and sort them
        # file_list = sorted([
        #     f for f, base_name in all_origin_files.items() 
        #     #if base_name in image_prompts
        # ])
        file_list = natsort.natsorted(all_origin_files)
    except FileNotFoundError:
        print(f"Error: Origin directory not found at {origin_dir}")
        return
    except Exception as e:
        print(f"Error listing files in origin directory: {e}")
        return

    if not file_list:
        print("No matching image files found in the origin directory.")
        return

    # Process each target directory
    for target_dir_path in target_dirs:
        dir_name = os.path.basename(os.path.normpath(target_dir_path))
        print(f"\nCalculating CLIP/Reward metrics for target directory: {target_dir_path}")

        clip_img_sims, clip_scores, reward_scores, reward_scores_origin, clip_scores_origin  = [], [], [], [], []
        
        all_target_files = os.listdir(target_dir_path) #{f: f.rsplit('.', 1)[0] for f in os.listdir(target_dir_path) if os.path.isfile(os.path.join(origin_dir, f))}
        #print(all_target_files)
        # Filter files to only those that match the base prompts and sort them
        names_origin = []
        names_target = []
        file_list_target = natsort.natsorted(all_target_files)
        for i, fname in tqdm(enumerate(file_list), desc=f"Processing {dir_name}"):
            # Get the base prompt name (e.g., 'prompt_0' from 'prompt_0.png')
            prompt_base_name = fname.rsplit('.', 1)[0]
            prompt_name = prompts[i]

            origin_img_path = os.path.join(origin_dir, fname)
            #target_img_path = os.path.join(target_dir_path, fname)
            
            target_img_path = os.path.join(target_dir_path, file_list_target[i])
            
            # print(target_img_path)
            # print(origin_img_path)
            # print("-" * 10)
            # Check if both files exist (the target directory must contain the same files)
            
            origin_img = Image.open(origin_img_path).convert("RGB")
            target_img = Image.open(target_img_path).convert("RGB")
            
            names_origin.append(fname)
            names_target.append( file_list_target[i])

            # Compute CLIP Embeddings
            origin_clip_feat = get_clip_image_embedding(origin_img)
            target_clip_feat = get_clip_image_embedding(target_img)
            
            # Compute Metrics
            clip_img_sim = compute_clip_image_score(origin_clip_feat, target_clip_feat)
            if clip_img_sim is not None: clip_img_sims.append(clip_img_sim)
            
            # Use the style prompt for the CLIP Score (Style Alignment)
            clip_score = compute_clip_score(target_img, style_prompt) 
            if clip_score is not None: clip_scores.append(clip_score)
            
            # Use the image's base prompt for the Image Reward model (Content Alignment/Quality)
            reward_score = compute_clip_score(target_img, prompt_name) 
            reward_scores.append(reward_score)


            clip_score_origin = compute_clip_score(origin_img, style_prompt) 
            if clip_score is not None: clip_scores_origin.append(clip_score_origin)
            
            # Use the image's base prompt for the Image Reward model (Content Alignment/Quality)
            reward_score_origin = compute_clip_score(origin_img, prompt_name) 
            reward_scores_origin.append(reward_score_origin)
    
    # 1. Calculate Mean Metrics
    summary_metrics = {
        "mean_clip_img_sim": float(np.mean(clip_img_sims)) if clip_img_sims else None,
        "mean_clip_score": float(np.mean(clip_scores)) if clip_scores else None,
        "mean_reward_score": float(np.mean(reward_scores)) if reward_scores else None,
        "mean_reward_score_origin": float(np.mean(reward_scores_origin)) if reward_scores_origin else None,
        "mean_clip_score_origin": float(np.mean(clip_scores_origin)) if clip_scores_origin else None
    }

    # 2. Create Per-Image Dictionary ("idx: metrics")
    per_image_metrics = {}
    for i in range(len(clip_img_sims)):
        # Using string 'i' because JSON keys must be strings
        per_image_metrics[str(i)] = {
            "clip_img_sim": clip_img_sims[i],
            "clip_score": clip_scores[i],
            "reward_score": reward_scores[i],
            "reward_score_origin": reward_scores_origin[i],
            "clip_score_origin": clip_scores_origin[i],
            "name_origin": names_origin[i],
            "name_target": names_target[i]
        }

    # 3. Save to Files
    # Ensure results_dir exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Save Summary
    print(summary_metrics)
    print("-" * 10)
    summary_path = os.path.join(args.save_dir, "summary_metrics.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)

    # Save Per-Image
    detailed_path = os.path.join(args.save_dir, "per_image_metrics.json")
    with open(detailed_path, 'w') as f:
        json.dump(per_image_metrics, f, indent=4)

    print(f"Metrics successfully saved to {args.save_dir}")
        
        # Aggregate Results
    #     results = {}
    #     results['clip_img_sim'] = np.mean(clip_img_sims) if clip_img_sims else None
    #     results['clip_score_vs_style'] = np.mean(clip_scores) if clip_scores else None
    #     results['image_reward_mock'] = np.mean(reward_scores) if reward_scores else None
    #     results['clip_score_vs_style_origin'] = np.mean(clip_scores_origin) if clip_scores_origin else None
    #     results['image_reward_mock_origin'] = np.mean(reward_scores_origin) if reward_scores_origin else None

    #     metrics_clip_reward[dir_name] = results

    # # Save Results
    # print(results)
    # os.makedirs(save_dir, exist_ok=True)
    # output_path = os.path.join(save_dir, SAVE_FILENAME)
    # torch.save(metrics_clip_reward, output_path)
    # print(f"\n✅ CLIP and Reward metrics computed and saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute CLIP and Reward Metrics.")
    parser.add_argument('--origin_dir', type=str, required=True, help="Path to the directory containing source/origin images.")
    parser.add_argument('--target_dirs', nargs='+', required=True, help="List of paths to directories containing target/stylized images.")
    parser.add_argument('--image_prompts', nargs='+', required=True, help="List of base image prompt names (e.g., 'prompt_0', 'prompt_1').")
    parser.add_argument('--style_prompt', type=str, required=True, help="The text prompt describing the target style (e.g., 'A watercolor painting').")
    parser.add_argument('--save_dir', type=str, default='.', help="Directory to save the resulting .pt file.")
    
    args = parser.parse_args()

    if clip_model:
        compute_clip_reward_metrics(args.origin_dir, args.target_dirs, args.image_prompts, args.style_prompt, args.save_dir)
    else:
        print("CLIP model not loaded. Skipping metric computation.")