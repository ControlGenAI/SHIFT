import os
import torch
from PIL import Image
import clip
from transformers import CLIPProcessor, CLIPModel
import natsort

# 1. Setup Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# class CLIPScorer:
#     def __init__(self):
#         # The SOTA model for COCO evaluation (ViT-g/14)
#         model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        
#         print(f"Initializing SOTA CLIP (ViT-g/14) on {DEVICE}...")
#         self.model = CLIPModel.from_pretrained(model_id).to(DEVICE)
#         self.processor = CLIPProcessor.from_pretrained(model_id)
#         self.model.eval()

#     @torch.no_grad()
#     def calculate_score(self, image_path, prompt_text):
#         image = Image.open(image_path).convert("RGB")
        
#         inputs = self.processor(
#             text=[prompt_text], 
#             images=image, 
#             return_tensors="pt", 
#             padding=True,
#             truncation=True
#         ).to(DEVICE)

#         outputs = self.model(**inputs)
        
#         # Normalize features
#         image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
#         text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        
#         # Cosine similarity * 100 (Standard scaling for papers)
#         # result of 0.30 -> 30.0
#         score = torch.matmul(image_embeds, text_embeds.t()).item() * 100
#         return score

class CLIPScorer:
    def __init__(self):
        print(f"Initializing CLIP (ViT-L/14) on {DEVICE}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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

# class CLIPScorer:
#     def __init__(self):
#         # Using the official OpenAI weights/logic
#         self.model, self.preprocess = clip.load("ViT-H/14", device=DEVICE)
#         self.model.eval()

#     @torch.no_grad()
#     def calculate_score(self, image_path, prompt_text):
#         # 1. Standardize the prompt
#         # Many papers use a template if the raw text is just a label
#         prompt = f"a photo of {prompt_text}" 
        
#         image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
#         text = clip.tokenize([prompt]).to(DEVICE)

#         # 2. Get features (model.encode_image/text handles projection and norm)
#         image_features = self.model.encode_image(image)
#         text_features = self.model.encode_text(text)

#         # 3. Normalize
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#         # 4. Calculate Cosine Similarity
#         similarity = (image_features @ text_features.T).item()
        
#         # 5. Scaling
#         # To get 30.0 instead of 0.30:
#         return similarity * 100

def run_test_on_dir(target_dir):
    scorer = CLIPScorer()
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)]
    image_files = natsort.natsorted(image_files)
    
    if not image_files:
        print(f"No images found in {target_dir}")
        return

    results = []
    print(f"\n--- Processing {len(image_files)} images ---\n")
    
    txt_path = '/home/jovyan/konovalova/steering/coco_10k_prompts.txt'

    with open(txt_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
            
    print(prompts)
    for i, filename in enumerate(image_files[:1800]):
        prompt = prompts[i]
        
        img_path = os.path.join(target_dir, filename)
        if i != int(filename.split('_')[0]):
            print(prompt, filename)
            assert False
            
        
        score = scorer.calculate_score(img_path, prompt)
        results.append(score)
        print(f"[✔] {filename}: {score:.4f}")
        
        

    if results:
        avg = sum(results) / len(results)
        print("\n" + "="*40)
        print(f"FINAL AVERAGE SCORE: {avg:.4f}")
        print("="*40)

# --- RUN ---
# Replace with your actual directory path
run_test_on_dir("/home/jovyan/konovalova/steering/experiments/flux_dev/remove/generated_images_big_dataset/coco/500_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_42/steered")

# run_test_on_dir("/home/jovyan/konovalova/steering/experiments/flux_dev/remove/generated_images_big_dataset/coco/1000_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2/steered")
# run_test_on_dir("/home/jovyan/konovalova/steering/generated_images")
