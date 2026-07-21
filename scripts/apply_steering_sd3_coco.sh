#!/bin/bash
# Step 3/3 (SD3.5): apply steering on COCO prompts (preservation / FID eval).
# Run after scripts/steering_calculate_sd3.sh
# Run from repository root: bash scripts/apply_steering_sd3_coco.sh
set -euo pipefail

DATA_DIR="experiments/sd35/remove/final_steering/block_steering/nudity"
RESULTS_DIR="experiments/sd35/remove/generated_images/block_diff/nudity_coco"
PROMPTS_PATH="prompts_collection/main_experiments/all_coco_prompts.txt"
SEEDS_PATH="prompts_collection/additiona_seeds/coco_seeds.txt"

mkdir -p "${RESULTS_DIR}"

CUDA_VISIBLE_DEVICES=0 python ./src/steering/apply_steering_with_injection_sd35.py \
    --model_name "stabilityai/stable-diffusion-3.5-medium" \
    --data_dir "${DATA_DIR}" \
    --task remove \
    --strength 50.0 \
    --strength_img 0.0 \
    --injection_point block \
    --vector_type diff \
    --steering_type separate \
    --steering_mode unconditional \
    --guidance_scale 4.5 \
    --inference_steps 28 \
    --width 512 \
    --height 512 \
    --steer_txt \
    --strength_txt 5.0 \
    --use_cls \
    --use_pooled_cosine_score \
    --results_dir "${RESULTS_DIR}" \
    --prompts_path "${PROMPTS_PATH}" \
    --seeds_path "${SEEDS_PATH}"
