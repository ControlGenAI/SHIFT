import torch
from diffusers import FluxPipeline
from datasets import load_dataset
import os
import pandas as pd
import torch
from diffusers import FluxPipeline
import os
from tqdm import tqdm

import torch
import copy
import random


torch.set_grad_enabled(False)

from safetensors.torch import load_file
from diffusers import DiffusionPipeline

import sys

torch_dtype = torch.bfloat16
device = 'cuda:0'

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", 
                                         torch_dtype=torch_dtype, 
                                         safety_checker=None).to(device)


output_dir = "/home/jovyan/konovalova/minimalist_concept_erasure/origin_coco"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv( '/home/jovyan/konovalova/steering/coco_30k.csv')
print(df.head(10))

# 4. Processing Loop
for index, row in tqdm(df.iterrows(), total=5000):
    prompt = row['prompt']
    seed = int(row['evaluation_seed'])
    coco_id = int(row['coco_id'])
    
    # Format filename to match COCO standard
    filename = f"{index}_{coco_id:012d}_flux.png"
    save_path = os.path.join(output_dir, filename)

    if os.path.exists(save_path):
        continue

    # Set the seed for exact reproducibility from the CSV
    generator = torch.Generator(device="cuda:0").manual_seed(seed)

    try:
        # Generate the image
        image = pipe(
            prompt=prompt,
            num_inference_steps=4,      # As requested
            guidance_scale=0.0,         # As requested
            generator=generator,
            width=512,                 # Default Flux resolution
            height=512
        ).images[0]

        # Save
        image.save(save_path)
        
    except Exception as e:
        print(f"Error generating image for ID {coco_id}: {e}")
    
    if index > 5000:
        assert False

# --- Execution ---
# run_flux_on_coco_data('coco_30k_10k.csv')
