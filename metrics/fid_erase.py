import os
import shutil
from pytorch_fid import fid_score
import os
import torch
import natsort
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from pytorch_fid import fid_score

DIR_PAIRS = [
    (
        "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/0_0_42/steered", 
        "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/0_0_10/steered"
    ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/0_0_10/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/10_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_not_normed_10/steered"
    # ),
    # ( "another_origin_path", "another_target_path" ),
]

CONCEPTS = ["Snoopy", "Spongebob", "Pikachu", "dog", "Mickey", "legislator"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_FILE = "prompts_collection/ablation/ablation_prompts_remove.txt"


def run_aggregated_fid(dir_pairs, concepts, device="cuda"):
    print("\n" + "="*50)
    print("STARTING AGGREGATED FID CALCULATION")
    print("="*50)

    # 1. Setup Temporary Directory Structure
    temp_root = "./temp_fid_aggregate"
    # Structure: temp_fid_aggregate/origin/{concept} and temp_fid_aggregate/target/{concept}
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    
    for side in ["origin", "target"]:
        for concept in concepts:
            os.makedirs(os.path.join(temp_root, side, concept), exist_ok=True)

    # 2. Aggregate Images across all directory pairs
    print("Grouping images by concept...")
    for orig_dir, target_dir in dir_pairs:
        # Process Origin side
        for fname in os.listdir(orig_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                for concept in concepts:
                    if concept.lower() in fname.lower():
                        # Use a prefix to prevent filename collisions across different pairs
                        unique_name = f"{os.path.basename(orig_dir)}_{fname}"
                        shutil.copy(os.path.join(orig_dir, fname), 
                                    os.path.join(temp_root, "origin", concept, unique_name))
        
        # Process Target (Steered) side
        for fname in os.listdir(target_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                for concept in concepts:
                    if concept.lower() in fname.lower():
                        unique_name = f"{os.path.basename(target_dir)}_{fname}"
                        shutil.copy(os.path.join(target_dir, fname), 
                                    os.path.join(temp_root, "target", concept, unique_name))

    # 3. Calculate FID per Concept
    fid_results = []
    for concept in concepts:
        path_origin = os.path.join(temp_root, "origin", concept)
        path_target = os.path.join(temp_root, "target", concept)
        
        # Check if we have images in both folders
        num_orig = len(os.listdir(path_origin))
        num_target = len(os.listdir(path_target))
        
        if num_orig > 1 and num_target > 1:
            print(f"\nCalculating FID for: {concept} ({num_orig} images)...")
            try:
                score = fid_score.calculate_fid_given_paths(
                    [path_origin, path_target],
                    batch_size=50,
                    device=device,
                    dims=2048
                )
                fid_results.append({"Concept": concept, "FID": f"{score:.2f}"})
            except Exception as e:
                print(f"Error calculating FID for {concept}: {e}")
                fid_results.append({"Concept": concept, "FID": "Error"})
        else:
            print(f"Skipping {concept}: Not enough images (Orig: {num_orig}, Target: {num_target})")
            fid_results.append({"Concept": concept, "FID": "N/A"})

    # 4. Cleanup and Display
    shutil.rmtree(temp_root)
    
    fid_df = pd.DataFrame(fid_results)
    print("\n" + "="*50)
    print("AGGREGATED FID RESULTS")
    print("="*50)
    print(fid_df.to_string(index=False))
    return fid_df

# To use it in your existing main block:
if __name__ == "__main__":
    # ... your existing CLIP code ...
    
    # Run the new FID function
    run_aggregated_fid(DIR_PAIRS, CONCEPTS, DEVICE)