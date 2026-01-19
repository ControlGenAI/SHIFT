import os
import json
import torch
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv

def calculate_iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea + 1e-6)

def main():
    parser = argparse.ArgumentParser(description="Native Grounding DINO Detection with Visualization")
    
    # Required Args
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of source images")
    parser.add_argument("--save_dir", type=str, required=True, help="Where to save JSON and visualized images")
    parser.add_argument("--target_prompt", type=str, required=True, help="Object to detect (e.g. 'glasses')")
    
    # DINO Thresholds
    parser.add_argument("--dino_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    
    # Visualization Flag
    parser.add_argument("--visualize", action="store_true", help="Save images with boxes and labels drawn")
    
    # Extra params
    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--gt_json", type=str, default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare output directories
    os.makedirs(args.save_dir, exist_ok=True)
    if args.visualize:
        vis_dir = os.path.join(args.save_dir, "visualized")
        os.makedirs(vis_dir, exist_ok=True)

    # 1. Load Model and Processor
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    # 2. Setup Annotators (Supervision)
    if args.visualize:
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

    # 3. Inference Loop
    results_dict = {}
    exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(exts)]
    
    # Add period for better DINO performance
    text_prompt = args.target_prompt if args.target_prompt.endswith(".") else f"{args.target_prompt}."

    print(f"Running Detection. Target: {text_prompt} | Visualize: {args.visualize}")

    for filename in tqdm(image_files):
        img_path = os.path.join(args.image_dir, filename)
        image_pil = Image.open(img_path).convert("RGB")
        
        inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.dino_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        # Save to main dict
        boxes_np = results["boxes"].cpu().numpy()
        scores_np = results["scores"].cpu().numpy()
        
        results_dict[img_path] = {
            'bbox_coordinates': boxes_np.tolist(),
            'scores': scores_np.tolist()
        }

        # 4. Visualization Logic
        if args.visualize:
            # Convert to Supervision Detections format
            if len(boxes_np) > 0:
                detections = sv.Detections(
                    xyxy=boxes_np,
                    confidence=scores_np,
                    class_id=np.zeros(len(boxes_np), dtype=int) # All are the target class
                )
                
                # Prepare labels
                labels = [f"{args.target_prompt} {conf:.2f}" for conf in scores_np]
                
                # Draw on image (Convert PIL to CV2 BGR)
                frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                
            # Save visualized image
                cv2.imwrite(os.path.join(vis_dir, filename), annotated_frame)
            else:
                image_pil.save(os.path.join(vis_dir, filename))

    # 5. Save Final JSON
    with open(os.path.join(args.save_dir, "detections.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)

    # 6. Calculate Metrics (Detection Rate)
    images_with_det = sum(1 for v in results_dict.values() if len(v['bbox_coordinates']) > 0)
    det_rate = images_with_det / len(image_files) if image_files else 0
    
    print(f"\n--- Evaluation ---")
    print(f"Total Images:   {len(image_files)}")
    print(f"Detection Rate: {det_rate:.2%}")
    
    with open(os.path.join(args.save_dir, "metrics.json"), 'w') as f:
        json.dump({"detection_rate": det_rate}, f, indent=4)

if __name__ == "__main__":
    main()