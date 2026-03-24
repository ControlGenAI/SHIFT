import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

def evaluate_erasure(csv_path, target_dir, target_artist, method_name="erased_model"):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # 2. Setup CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # 3. Define Classification Labels (Artists + Neutral)
    unique_artists = df['artist'].unique().tolist()
    # Adding neutral labels to see if the erased style "drops" into a generic category
    neutral_labels = ["a photo", "a generic digital image", "an image with no specific artistic style"]
    all_labels = unique_artists + neutral_labels
    
    # Pre-encode text labels for classification
    with torch.no_grad():
        inputs = processor(text=all_labels, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    results = []

    print(f"Processing images in {target_dir}...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        case_num = row['case_number']
        img_path = os.path.join(target_dir, f"{case_num}.png")
        
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        
        # Prepare inputs for Score Calculation
        # We check similarity against the specific artist name and the full prompt
        inputs = processor(text=[row['artist'], row['prompt']], images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Image and text features
            image_embed = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # CLIP Scores (Cosine Similarity * 100)
            # index 0 is artist, index 1 is prompt
            score_artist = (image_embed @ text_embeds[0].T).item() * 100
            score_prompt = (image_embed @ text_embeds[1].T).item() * 100
            
            # CLIP Classification
            logits = (image_embed @ text_features.T) * model.logit_scale.exp()
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
            pred_idx = probs.argmax()
            prediction = all_labels[pred_idx]
            is_neutral = prediction in neutral_labels

        results.append({
            'case_number': case_num,
            'artist': row['artist'],
            'clip_score_artist': score_artist,
            'clip_score_prompt': score_prompt,
            'clip_prediction': prediction,
            'is_neutral_pred': is_neutral,
            'target_prob': probs[all_labels.index(target_artist)] if target_artist in all_labels else 0
        })

    # 4. Save detailed results
    res_df = pd.DataFrame(results)
    output_filename = f"eval_results_{method_name}.csv"
    res_df.to_csv(output_filename, index=False)
    
    # 5. Calculate Summary Metrics
    print(f"\n--- Summary for Method: {method_name} ---")
    
    # Target Concept Metrics
    target_mask = res_df['artist'] == target_artist
    target_metrics = res_df[target_mask]
    
    # Other Concepts Metrics
    other_metrics = res_df[~target_mask]
    
    mean_target_score = target_metrics['clip_score_artist'].mean()
    mean_other_scores = other_metrics.groupby('artist')['clip_score_artist'].mean()
    total_mean_others = mean_other_scores.mean()
    
    # Neutrality (How often target is classified as neutral)
    neutral_rate_target = target_metrics['is_neutral_pred'].mean() * 100

    print(f"Target Artist: {target_artist}")
    print(f"Mean CLIP Score (Target): {mean_target_score:.2f}")
    print(f"Neutral Classification Rate (Target): {neutral_rate_target:.2f}%")
    print("-" * 30)
    print(f"Mean CLIP Score (Others - Avg of 4): {total_mean_others:.2f}")
    for artist, score in mean_other_scores.items():
        print(f"  - {artist}: {score:.2f}")
        
    return res_df

# Usage
evaluate_erasure('big_artist_prompts.csv', 'path/to/images', 'Pablo Picasso', 'MyErasureMethod')