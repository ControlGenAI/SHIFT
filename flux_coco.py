# import pandas as pd
# import torch
# from diffusers import FluxPipeline
# import os
# from tqdm import tqdm


#     # 2. Load the CSV
# df = pd.read_csv( '/home/jovyan/konovalova/steering/coco_30k.csv')
# print(f"Loaded {len(df)} prompts from")

# # 3. Initialize Flux-dev Pipeline
# # Note: Requires 'accelerate' and 'sentencepiece' libraries
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev", 
#     torch_dtype=torch.bfloat16
# )

# # Move to GPU
# pipe.to("cuda:0")

# # Optional: Enable memory efficient attention if you have lower VRAM
# # pipe.enable_model_cpu_offload() 

# output_dir="/home/jovyan/konovalova/steering/flux_generated_images_coco_correct_512_512_1"


# os.makedirs(output_dir, exist_ok=True)


# # 4. Processing Loop
# for index, row in tqdm(df.iterrows(), total=10000):
#     prompt = row['prompt']
#     seed = int(row['evaluation_seed'])
#     coco_id = int(row['coco_id'])
    
#     # Format filename to match COCO standard
#     filename = f"{index}_{coco_id:012d}_flux.png"
#     save_path = os.path.join(output_dir, filename)

#     if os.path.exists(save_path) or index < 0 :
#         continue

#     # Set the seed for exact reproducibility from the CSV
#     generator = torch.Generator(device="cuda").manual_seed(seed)

#     try:
#         # Generate the image
#         image = pipe(
#             prompt=prompt,
#             num_inference_steps=28,      # As requested
#             guidance_scale=1.0,         # As requested
#             generator=generator,
#             width=512,                 # Default Flux resolution
#             height=512,
#             max_sequence_length=512
#         ).images[0]

#         # Save
#         image.save(save_path)
        
#     except Exception as e:
#         print(f"Error generating image for ID {coco_id}: {e}")

#     if index >= 10000:
#         break


# # --- Execution ---
# # run_flux_on_coco_data('coco_30k_10k.csv')


import pandas as pd
import torch
from diffusers import FluxPipeline
import os
from tqdm import tqdm
import random

# --- CONFIGURATION ---
csv_path = '/home/jovyan/konovalova/steering/coco_30k.csv'
output_dir = "/home/jovyan/konovalova/steering/flux_generated_images_coco_10k"
seed_log_path = os.path.join("/home/jovyan/konovalova/steering/", "generation_seeds.txt")
num_samples = 10000

os.makedirs(output_dir, exist_ok=True)

# 1. Load and Randomly Subsample
df_full = pd.read_csv(csv_path)
# Randomly pick 10k rows
df = df_full.sample(n=num_samples, random_state=42).reset_index(drop=True) 
print(f"Sampled {len(df)} prompts from original CSV.")
df.to_csv('coco_10k_seed42.csv', index=False)

# 2. Initialize Flux-dev
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
)
pipe.to("cuda:0")

# 3. Open seed log file for writing
with open(seed_log_path, "w") as f_seed:
    f_seed.write("index,coco_id,seed\n") # Header

    # 4. Processing Loop
    for index, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        coco_id = int(row['coco_id'])
        
        # Generate a new random seed for this run
        current_seed = random.randint(0, 10**6)
        
        filename = f"{index}_{coco_id:012d}_flux.png"
        save_path = os.path.join(output_dir, filename)

        # Log the seed immediately (in case of crash)
        f_seed.write(f"{index},{coco_id},{current_seed}\n")
        f_seed.flush() 

        if os.path.exists(save_path):
            continue

        generator = torch.Generator(device="cuda").manual_seed(current_seed)

        try:
            image = pipe(
                prompt=prompt,
                num_inference_steps=28,
                guidance_scale=3.5, # Recommendation: Use 3.5 for Flux-dev FID
                generator=generator,
                width=512,
                height=512,
                max_sequence_length=512
            ).images[0]

            image.save(save_path)
            
        except Exception as e:
            print(f"\nError generating image for ID {coco_id}: {e}")

        if index > 1000:
            break

print(f"Finished! Seeds saved to {seed_log_path}")