"""CLIP score (ViT-L/14, x100) for the concrete-concept erasure protocol.

Image files are expected as "<prompt_idx>_..._<concept>_...png"; the text is
"<prompt[prompt_idx]> <concept>". Scores are averaged per concept over all
target directories (e.g. one directory per seed).

Usage:
    python metrics/clip_erase.py --target_dirs <seed_10_dir> <seed_20_dir> ... \
        --concepts Snoopy Mickey Spongebob Pikachu Dog Legislator \
        --prompts_path prompts_collection/ablation/ablation_prompts_remove.txt
"""
import argparse
import os
import torch
import natsort
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CASteerEvaluator:
    def __init__(self, concepts, prompt_file, model_id="openai/clip-vit-large-patch14"):
        print(f"Initializing CLIP ({model_id}) on {DEVICE}...")
        self.model = CLIPModel.from_pretrained(model_id).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()
        self.concepts = concepts

        # Pre-load prompts to avoid re-reading files in loops
        with open(prompt_file, "r") as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def get_concept_from_filename(self, fname):
        for concept in self.concepts:
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

    def evaluate_directory(self, directory):
        """Helper to process a single directory and return scores per concept."""
        files = natsort.natsorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg'))])
        scores_per_concept = {c: [] for c in self.concepts}

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

    def run_evaluation(self, target_dir):
        """Runs evaluation for one directory and returns a dataframe of results."""
        target_results = self.evaluate_directory(target_dir)

        pair_data = []
        for concept in self.concepts:
            steer_mean = np.mean(target_results[concept]) if target_results[concept] else 0
            pair_data.append({
                "Concept": concept,
                "Steered_CS": steer_mean
            })

        return pd.DataFrame(pair_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP score for concrete concept erasure (target + preserved concepts).")
    parser.add_argument("--target_dirs", nargs="+", required=True, help="Directories with generated images (e.g. one per seed).")
    parser.add_argument("--concepts", nargs="+", default=["Snoopy", "Mickey", "Spongebob", "Pikachu", "Dog", "Legislator"],
                        help="Concept names; matched case-insensitively against file names.")
    parser.add_argument("--prompts_path", type=str, default="prompts_collection/ablation/ablation_prompts_remove.txt")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    args = parser.parse_args()

    evaluator = CASteerEvaluator(args.concepts, args.prompts_path, args.clip_model)
    all_dfs = []

    for i, target in enumerate(args.target_dirs):
        print(f"\n--- Evaluating directory {i+1}/{len(args.target_dirs)} ---")
        print(f"Target: {target}")

        df_pair = evaluator.run_evaluation(target)
        all_dfs.append(df_pair)

    # Combine all results
    if all_dfs:
        full_results = pd.concat(all_dfs)

        # Calculate Mean across all directories and concepts
        mean_steered = full_results["Steered_CS"].mean()

        # Concept-wise summary
        summary = full_results.groupby("Concept").mean().reset_index()

        print("\n" + "="*50)
        print("FINAL CONSOLIDATED RESULTS (Concept-wise Mean)")
        print("="*50)
        print(summary.to_string(index=False))

        print("\n" + "="*50)
        print(f"GRAND TOTAL MEAN STEERED CS: {mean_steered:.2f}")
        print("="*50)
