import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import argparse
import natsort


# --- Configuration & Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILENAME = 'metrics_dinov2.pt'

dinov2 = None
dinov2_transform = T.Compose([
    T.Resize(224, interpolation=Image.Resampling.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    print("Initializing DINOv2 Model...")
    # Load DINOv2 model (ViT-L/14)
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    dinov2.eval()
except Exception as e:
    print(f"Error loading DINOv2: {e}. DINOv2 metrics will be skipped.")

# --- DINOv2 Metric Functions ---

def get_dinov2_embedding(img: Image.Image) -> torch.Tensor:
    """Gets the normalized DINOv2 feature vector (flattened)."""
    if dinov2 is None: return None
    with torch.no_grad():
        img_t = dinov2_transform(img).unsqueeze(0).to(device)
        feats = dinov2(img_t).flatten()
        return feats / feats.norm()

def compute_dinov2_image_score(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
    """Computes DINOv2 Image-to-Image Cosine Similarity from normalized features."""
    if feat1 is None or feat2 is None: return None
    return torch.nn.functional.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

# --- Main Logic ---

def compute_dino_metrics(origin_dir, target_dirs, image_prompts, save_dir):
    """
    Computes DINOv2 Image-to-Image similarity between origin and target images.
    """
    if not dinov2:
        print("DINOv2 model not available. Skipping metric computation.")
        return

    metrics_dinov2 = {}

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
        # Assuming all origin images are listed in the provided image_prompts list
        # We find files in the origin_dir that match the prompts
        all_origin_files = {f: f.rsplit('.', 1)[0] for f in os.listdir(origin_dir) if os.path.isfile(os.path.join(origin_dir, f))}
        
        # Filter files to only those that match the base prompts and sort them
        # file_list = sorted([
        #     f for f, base_name in all_origin_files.items() 
        #     # if base_name in image_prompts
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
        print(f"\nCalculating DINOv2 similarity for target directory: {dir_name}")
        
        dinov2_sims = []
        
        for i, fname in tqdm(enumerate(file_list), desc=f"Processing {dir_name}"):
            # Get the base prompt name (e.g., 'prompt_0' from 'prompt_0.png')
            prompt_base_name = fname.rsplit('.', 1)[0]

            origin_img_path = os.path.join(origin_dir, fname)
            prompt_name = prompts[i]
            target_img_path = os.path.join(target_dir_path, prompt_name + '.png')
            
            # Check if both files exist (the target directory must contain the same files)
            if os.path.exists(target_img_path):
                try:
                    origin_img = Image.open(origin_img_path).convert("RGB")
                    target_img = Image.open(target_img_path).convert("RGB")
                except Exception as e:
                    print(f"Could not load image {fname} in {target_dir_path}: {e}")
                    continue
            else:
                # Target file is missing, skip this pair
                continue

            # Compute DINOv2 Metrics
            origin_dinov2_feat = get_dinov2_embedding(origin_img)
            target_dinov2_feat = get_dinov2_embedding(target_img)
            
            dinov2_sim = compute_dinov2_image_score(origin_dinov2_feat, target_dinov2_feat)
            if dinov2_sim is not None: 
                dinov2_sims.append(dinov2_sim)
        
        # Aggregate Results
        metrics_dinov2[dir_name] = {'dinov2_img_sim': np.mean(dinov2_sims) if dinov2_sims else None}

    # Save Results
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, SAVE_FILENAME)
    torch.save(metrics_dinov2, output_path)
    print(f"\n✅ DINOv2 metrics computed and saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute DINOv2 Image Similarity Metrics.")
    parser.add_argument('--origin_dir', type=str, required=True, help="Path to the directory containing source/origin images.")
    parser.add_argument('--target_dirs', nargs='+', required=True, help="List of paths to directories containing target/stylized images.")
    # The image_prompts argument is still used to FILTER files, but loading uses os.listdir/sorted
    parser.add_argument('--image_prompts', nargs='+', required=True, help="List of base image prompt names (e.g., 'prompt_0', 'prompt_1').") 
    parser.add_argument('--save_dir', type=str, default='.', help="Directory to save the resulting .pt file.")
    parser.add_argument('--style_prompt', type=str, default="", help="Style description prompt (not used for DINOv2 img sim).") 
    
    args = parser.parse_args()
    compute_dino_metrics(args.origin_dir, args.target_dirs, args.image_prompts, args.save_dir)