import torch
import random
import os
from diffusers import FluxPipeline
from datasets import load_dataset
from tqdm import tqdm

# 1. Setup Model (FLUX.1-schnell)
model_id = "black-forest-labs/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() # Use this for 24GB VRAM cards. 

# 2. Load COCO Captions (Validation Set)
print("Loading MS-COCO dataset...")
dataset = load_dataset("detection-datasets/coco", split="val")
all_captions = [item['caption'] for item in dataset]

# 3. Sample 1,000 Random Prompts
random.seed(42) # For reproducibility
sampled_prompts = random.sample(all_captions, 1000)

# 4. Save Prompts to TXT File
with open("coco_prompts_1000.txt", "w", encoding="utf-8") as f:
    for prompt in sampled_prompts:
        f.write(f"{prompt}\n")
print("Saved 1,000 prompts to coco_prompts_1000.txt")

# 5. Image Generation Loop
output_dir = "./flux_coco_1000"
os.makedirs(output_dir, exist_ok=True)

print("Starting generation...")
for i, prompt in enumerate(tqdm(sampled_prompts)):
    # Schnell is a distilled model: 4 steps and 0.0 guidance are standard
    image = pipe(
        prompt,
        num_inference_steps=4, 
        guidance_scale=0.0,
        generator=torch.Generator("cpu").manual_seed(i)
    ).images[0]
    
    image.save(f"{output_dir}/img_{i:04d}.png")

print(f"Done! Images saved to {output_dir}")