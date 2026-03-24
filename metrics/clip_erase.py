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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
import torch
import natsort
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# --- Configuration ---
# List of (origin_dir, target_dir) tuples
DIR_PAIRS = [
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/10_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_not_normed_42/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/1000_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_all_42/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_no_cls/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_m_one_vec_new_only_pooled_2_scaled_clip/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_new_only_pooled_2_scaled_clip/steered"
    # ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_42/steered", 
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_normals_s_one_vec_new_only_pooled_2_scaled_clip/steered"
    ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/0_0_10/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/10_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_not_normed_10/steered"
    # ),
    # ( "another_origin_path", "another_target_path" ),
    # (
    #     "/home/jovyan/konovalova/erasing/notebooks/origin_big", 
    #     "/home/jovyan/konovalova/erasing/notebooks/big_artists"
    # ),
]

# DIR_PAIRS = [
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_10/steered", 
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_10"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_20/steered", 
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_20"
#     ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_30/steered", 
    #     "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_30"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_42/steered", 
    #     "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_42"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_50/steered", 
    #     "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_50"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_60/steered", 
    #     "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_60"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_70/steered", 
    #     "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_70"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_80/steered", 
    #     "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_80"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_90/steered", 
    #     "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_90"
    # ),
    
# ]

# DIR_PAIRS = [
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_10/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_10"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_20/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_20"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_30/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_30"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_42/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_42"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_50/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_50"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_60/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_60"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_70/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_70"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_80/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_80"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_90/steered", 
#         "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_90"
#     ),
    
# ]



# DIR_PAIRS = [
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_10/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_10/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_20/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_20/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_30/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_30/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_42/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_42/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_50/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_50/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_60/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_60/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_70/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_70/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_80/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_80/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_90/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_90/steered"
#     ),
    
# ]

# DIR_PAIRS = [
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_10",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_10"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_20",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_20"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_30",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_30"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_42",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_42"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_50",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_50"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_60",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_60"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_70",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_70"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_80",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_80"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_90",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_90"
#     ),
    
# ]

DIR_PAIRS = [
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_10/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_10"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_20/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_20"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_30/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_30"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_42/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_42"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_50/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_50"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_60/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_60"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_70/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_70"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_80/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_80"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_90/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_90"
    ),
    
]

CONCEPTS = ["Pikachu"]
# CONCEPTS = ["Warhol", "Gogh", "Picasso", "Rembrandt", "Caravaggio"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_FILE = "prompts_collection/ablation/ablation_prompts_remove.txt"

class CASteerEvaluator:
    def __init__(self):
        print(f"Initializing CLIP (ViT-L/14) on {DEVICE}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        
        # Pre-load prompts to avoid re-reading files in loops
        with open(PROMPT_FILE, "r") as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def get_concept_from_filename(self, fname):
        for concept in CONCEPTS:
            if concept.lower() in fname.lower():
                return concept
        return None

    def compute_clip_score(self, image: Image.Image, text: str):
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_feat = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_feat = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = (img_feat * txt_feat).sum(dim=1).item()
        return similarity * 100 

    def evaluate_directory(self, directory, is_target=True):
        """Helper to process a single directory and return scores per concept."""
        files = natsort.natsorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg'))])
        scores_per_concept = {c: [] for c in CONCEPTS}

        for fname in tqdm(files, desc=f"Processing {os.path.basename(directory)}", leave=False):
            concept = self.get_concept_from_filename(fname)
            if concept:
                try:
                    num = int(fname.split('_')[0])
                    prompt = f"{self.prompts[num]} {concept}"
                    img = Image.open(os.path.join(directory, fname)).convert("RGB")
                    score = self.compute_clip_score(img, prompt)
                    scores_per_concept[concept].append(score)
                except (ValueError, IndexError):
                    continue
        return scores_per_concept

    def run_evaluation(self, origin_dir, target_dir):
        """Runs evaluation for one pair and returns a dataframe of results."""
        target_results = self.evaluate_directory(target_dir)
        #origin_results = self.evaluate_directory(origin_dir)

        pair_data = []
        for concept in CONCEPTS:
            #orig_mean = np.mean(origin_results[concept]) if origin_results[concept] else 0
            steer_mean = np.mean(target_results[concept]) if target_results[concept] else 0
            pair_data.append({
                "Concept": concept,
               # "Origin_CS": orig_mean,
                "Steered_CS": steer_mean
            })
        
        return pd.DataFrame(pair_data)

if __name__ == "__main__":
    evaluator = CASteerEvaluator()
    all_dfs = []

    for i, (orig, target) in enumerate(DIR_PAIRS):
        print(f"\n--- Evaluating Pair {i+1}/{len(DIR_PAIRS)} ---")
        print(f"Origin: {orig}\nTarget: {target}")
        
        df_pair = evaluator.run_evaluation(orig, target)
        all_dfs.append(df_pair)

    # Combine all results
    if all_dfs:
        full_results = pd.concat(all_dfs)
        
        # Calculate Mean across all directory pairs and concepts
        #mean_origin = full_results["Origin_CS"].mean()
        mean_steered = full_results["Steered_CS"].mean()
        
        # Concept-wise summary
        summary = full_results.groupby("Concept").mean().reset_index()

        print("\n" + "="*50)
        print("FINAL CONSOLIDATED RESULTS (Concept-wise Mean)")
        print("="*50)
        print(summary.to_string(index=False))
        
        print("\n" + "="*50)
        #print(f"GRAND TOTAL MEAN ORIGIN CS:  {mean_origin:.2f}")
        print(f"GRAND TOTAL MEAN STEERED CS: {mean_steered:.2f}")
        print("="*50)