import torch
from diffusers import FluxPipeline
import os

# Load model (adapt based on your base model, e.g., Flux)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.to('cuda')  # For GPU efficiency

# Output directory
os.makedirs('generated_images', exist_ok=True)

# Load captions
with open('coco_10k_prompts.txt', 'r') as f:
    captions = [line.strip() for line in f]

# Generate images (batch size 1 for simplicity; increase if GPU allows)
for i, caption in enumerate(captions):
    generator = torch.Generator().manual_seed(42)
    image = pipe(
        caption, 
        num_inference_steps=4, 
        guidance_scale=0.0,
        generator=generator
    ).images[0]
    image.save(f'generated_images/image_{i:05d}.png')
    print(f"Generated {i+1}/10000")

print("Generation complete.")