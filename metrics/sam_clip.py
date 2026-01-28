import os
import json
import torch
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
import supervision as sv

def main():
    parser = argparse.ArgumentParser(description="SAM + CLIP Zero-Shot Detection")
    
    # Required Args
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of source images")
    parser.add_argument("--save_dir", type=str, required=True, help="Where to save results")
    parser.add_argument("--target_prompt", type=str, required=True, help="Object to detect (e.g. 'glasses')")
    
    # Thresholds
    parser.add_argument("--clip_threshold", type=float, default=0.25, help="CLIP similarity threshold")
    
    # Models
    parser.add_argument("--sam_model", type=str, default="facebook/sam-vit-base", help="SAM model ID")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model ID")
    
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Models
    print(f"Loading SAM ({args.sam_model}) and CLIP ({args.clip_model})...")
    mask_generator = pipeline("mask-generation", model=args.sam_model, device=0 if device == "cuda" else -1)
    
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    # 2. Setup Directories
    os.makedirs(args.save_dir, exist_ok=True)
    if args.visualize:
        vis_dir = os.path.join(args.save_dir, f"vis_sam_clip_{args.clip_threshold}")
        os.makedirs(vis_dir, exist_ok=True)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

    # 3. Process Images
    results_dict = {}
    exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(exts)]

    for filename in tqdm(image_files):
        img_path = os.path.join(args.image_dir, filename)
        image_pil = Image.open(img_path).convert("RGB")
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # A. SAM: Generate all possible object masks
        sam_result = mask_generator(image_pil, points_per_batch=64)
        masks = sam_result["masks"]
        
        if not masks:
            results_dict[img_path] = {'bbox_coordinates': [], 'scores': []}
            continue

        # B. CLIP: Score each mask crop
        crops = []
        bboxes = [] # Format: [x1, y1, x2, y2]
        
        for mask in masks:
            if not np.any(mask):
                continue
            
            # Calculate bounding box from binary mask
            y_indices, x_indices = np.where(mask)
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            
            if x2 <= x1 or y2 <= y1:
                continue

            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            crop = image_pil.crop((x1, y1, x2, y2))
            crops.append(crop)

        if not crops:
            results_dict[img_path] = {'bbox_coordinates': [], 'scores': []}
            continue

        inputs = clip_processor(
            text=[f"a photo of {args.target_prompt}", "a photo of something else"],
            images=crops,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            target_scores = probs[:, 0].cpu().numpy()

        # C. Find only the 1 Best Match
        best_idx = np.argmax(target_scores)
        best_score = target_scores[best_idx]
        
        # Apply threshold to the best match
        if best_score > args.clip_threshold:
            final_boxes = np.array([bboxes[best_idx]])
            final_scores = np.array([best_score])
        else:
            final_boxes = np.array([])
            final_scores = np.array([])

        results_dict[img_path] = {
            'bbox_coordinates': final_boxes.tolist(),
            'scores': final_scores.tolist()
        }

        # D. Visualization (Single Best Match)
        if args.visualize:
            if len(final_boxes) > 0:
                detections = sv.Detections(
                    xyxy=final_boxes.astype(np.float32),
                    confidence=final_scores.astype(np.float32),
                    class_id=np.zeros(len(final_boxes), dtype=int)
                )
                labels = [f"{args.target_prompt} {final_scores[0]:.2f}"]
                
                annotated_frame = box_annotator.annotate(scene=image_cv2.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                cv2.imwrite(os.path.join(vis_dir, filename), annotated_frame)
            else:
                cv2.imwrite(os.path.join(vis_dir, filename), image_cv2)

    # 4. Save Final JSON
    with open(os.path.join(args.save_dir, "detections.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"\nDone! Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()