"""CLIP score on MS-COCO generations (prompt i <-> image whose name starts with "i_").

Usage:
    python metrics/clip_coco.py --image_dir <generated_dir> \
        --prompts_path prompts_collection/main_experiments/all_coco_prompts.txt
"""
import argparse
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import natsort

# 1. Setup Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPScorer:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        print(f"Initializing CLIP ({model_id}) on {DEVICE}...")
        self.model = CLIPModel.from_pretrained(model_id).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def calculate_score(self, image_path, prompt_text):
        image = Image.open(image_path).convert("RGB")

        # Process inputs
        inputs = self.processor(
            text=[prompt_text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        # Get features
        outputs = self.model(**inputs)

        # Standard CLIP Score calculation:
        # Normalize the features and calculate cosine similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity
        score = torch.matmul(image_embeds, text_embeds.t()).item()
        return score

def run_test_on_dir(target_dir, txt_path, max_images=None, model_id="openai/clip-vit-base-patch32"):
    scorer = CLIPScorer(model_id)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)]
    image_files = natsort.natsorted(image_files)

    if not image_files:
        print(f"No images found in {target_dir}")
        return

    if max_images is not None:
        image_files = image_files[:max_images]

    results = []
    print(f"\n--- Processing {len(image_files)} images ---\n")

    with open(txt_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    for i, filename in enumerate(image_files):
        prompt = prompts[i]

        img_path = os.path.join(target_dir, filename)
        if i != int(filename.split('_')[0]):
            raise ValueError(f"Image/prompt index mismatch at {i}: {filename!r} vs prompt {prompt!r}")

        score = 100 * scorer.calculate_score(img_path, prompt)  # x100, as reported in the paper
        results.append(score)
        print(f"[✔] {filename}: {score:.4f}")

    if results:
        avg = sum(results) / len(results)
        print("\n" + "="*40)
        print(f"FINAL AVERAGE CLIP SCORE (x100): {avg:.2f}")
        print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP score of generated MS-COCO images against their captions.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with generated images named '<idx>_*.png'.")
    parser.add_argument("--prompts_path", type=str, default="prompts_collection/main_experiments/all_coco_prompts.txt",
                        help="Caption file, one prompt per line, in generation order.")
    parser.add_argument("--max_images", type=int, default=None, help="Evaluate only the first N images (default: all).")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    run_test_on_dir(args.image_dir, args.prompts_path, args.max_images, args.clip_model)
