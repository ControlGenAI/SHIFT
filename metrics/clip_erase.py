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

# --- Configuration ---
ORIGIN_DIR = "experiments/flux_schnell/remove/generated_images/snoopy/0_0/steered"
TARGET_DIR = "experiments/flux_schnell/remove/generated_images/snoopy/800_20_mean_diff_s_only_pooled_one_vec_0_7/steered"
CONCEPTS = ["Snoopy", "Spongebob", "Mickey"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Configuration ---

SAVE_PATH = "./evaluation_results.json"


class CASteerEvaluator:
    def __init__(self):
        print(f"Initializing CLIP (ViT-L/14) on {DEVICE}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()

    def get_concept_from_filename(self, fname):
        """Extracts target concept based on keywords in your filename format."""
        for concept in CONCEPTS:
            if concept.lower() in fname.lower():
                return concept
        return None

    def compute_clip_score(self, image: Image.Image, text: str):
        """Calculates 100 * Cosine Similarity."""
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Normalized embeddings
            img_feat = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_feat = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = (img_feat * txt_feat).sum(dim=1).item()
        return similarity * 100 

    def run_evaluation(self, origin_dir, target_dir):
        print(target_dir)
        files = natsort.natsorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg'))])
        
        results = {c: {"clip_scores_steered": [], "clip_scores_origin": [], "fid": 0.0} for c in CONCEPTS}
        with open("prompts_collection/ablation/ablation_prompts_remove.txt", "r") as f:
            prompts = [line.strip() for line in f if line.strip()][:29]
        #1. Calculate CLIP Scores
        print("\n--- Calculating CLIP Similarity ---")
        # print(files)
        for fname in tqdm(files):
            concept = self.get_concept_from_filename(fname)
            if concept:
                num = int(fname.split('_')[0])
                prompt = prompts[num]
                img_path = os.path.join(target_dir, fname)
                img = Image.open(img_path).convert("RGB")
                prompt = f"{prompt} {concept}"
                print(prompt)
                score = self.compute_clip_score(img, prompt)
                results[concept]["clip_scores_steered"].append(score)

        files = natsort.natsorted([f for f in os.listdir(origin_dir) if f.lower().endswith(('.png', '.jpg'))])
        
        # 1. Calculate CLIP Scores
        print("\n--- Calculating CLIP Similarity ---")
        print(origin_dir)
        for fname in tqdm(files):
            concept = self.get_concept_from_filename(fname)
            if concept:
                num = int(fname.split('_')[0])
                prompt = prompts[num]
                
                img_path = os.path.join(origin_dir, fname)
                img = Image.open(img_path).convert("RGB")
                prompt = f"{prompt} {concept}"
                print(prompt)
                score = self.compute_clip_score(img, prompt)
                results[concept]["clip_scores_origin"].append(score)

        # print(results.keys())

        # 2. Calculate FID per Concept
        # FID requires comparing two full distributions. We create temp folders to isolate concepts.
        print("\n--- Calculating FID Scores ---")
        temp_root = "./temp_fid_eval"
        os.makedirs(temp_root, exist_ok=True)

        # for concept in CONCEPTS:
        #     tmp_origin = os.path.join(temp_root, f"{concept}_origin")
        #     tmp_target = os.path.join(temp_root, f"{concept}_target")
        #     os.makedirs(tmp_origin, exist_ok=True)
        #     os.makedirs(tmp_target, exist_ok=True)

        #     # Gather concept-specific images from both directories
        #     count = 0
        #     for d, tmp in [(origin_dir, tmp_origin), (target_dir, tmp_target)]:
        #         print(d)
                
        #         for f in os.listdir(d):
                    
        #             if concept.lower() in f.lower():
        #                 shutil.copy(os.path.join(d, f), os.path.join(tmp, f))
        #                 if d == target_dir: count += 1

        #     if count < 2:
        #         print(f"Skipping FID for {concept}: Insufficient images.")
        #         results[concept]["fid"] = float('nan')
        #     else:
        #         score = fid_score.calculate_fid_given_paths(
        #             [tmp_origin, tmp_target], batch_size=50, device=DEVICE, dims=2048
        #         )
        #         results[concept]["fid"] = score

        # # Cleanup temp folders
        # shutil.rmtree(temp_root)

        # 3. Finalize Data
        final_table = []
        for concept, data in results.items():
            mean_cs = np.mean(data["clip_scores_origin"]) if data["clip_scores_origin"] else 0
            mean_cs_target = np.mean(data["clip_scores_steered"]) if data["clip_scores_steered"] else 0
            final_table.append({
                "Concept": concept,
                "CS (CLIP Similarity)": f"{mean_cs:.2f}",
                "CS (CLIP Similarity) steered": f"{mean_cs_target:.2f}",
                "FID": f"{data['fid']:.2f}"
            })
        
        df = pd.DataFrame(final_table)
        print("\nEvaluation Results (Similar to CASteer Table 15):")
        print(df.to_string(index=False))
        #df.to_json(SAVE_PATH, orient="records")

if __name__ == "__main__":
    evaluator = CASteerEvaluator()
    evaluator.run_evaluation(ORIGIN_DIR, TARGET_DIR)