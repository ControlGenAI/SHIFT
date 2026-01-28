import torch
from diffusers import FluxPipeline
from datasets import load_dataset
import os

# 1. Setup Model
model_id = "black-forest-labs/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="balanced")


# 2. Load Dataset (Nudity/Sexual subset)
dataset = load_dataset("AIML-TUDA/i2p", split="train")

os.makedirs("flux_i2p_results", exist_ok=True)

from tqdm import tqdm
# 3. Generation Loop using Seeds from Dataset
for i, item in tqdm(enumerate(dataset)):
    prompt = item['prompt']
    # Extract the seed from the dataset record
    # Note: I2P typically uses the key 'sd_seed' or 'seed'
    target_seed = item['sd_seed'] 
    
    if os.path.exists(f"flux_i2p_results/idx_{i}_seed_{target_seed}.png"):
        print(f"Skipping index {i}: File already exists.")
        continue
    
    # Initialize generator with the specific seed
    generator = torch.Generator(device="cuda").manual_seed(target_seed)
    
    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0, # Flux Schnell requirement
        generator=generator,
        height=512,
        width=512
    ).images[0]
    
    # Save file with the seed in the filename for verification
    image.save(f"flux_i2p_results/idx_{i}_seed_{target_seed}.png")
    
    if i % 10 == 0:
        print(f"Processed {i}/{len(dataset)} prompts using dataset seeds.")