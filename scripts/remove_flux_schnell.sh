#!/bin/bash
# Step 3 (FLUX.1-schnell): apply steering for concept removal / nudity erase.
# Edit DATA_DIR / RESULTS_DIR / PROMPTS_PATH before running.
# Run from repository root: bash scripts/remove_flux_schnell.sh
set -euo pipefail

DATA_DIR="experiments/flux_schnell/remove/final_steering/block_steering/nudity"
RESULTS_DIR="experiments/flux_schnell/remove/generated_images/nudity"
PROMPTS_PATH="prompts_collection/main_experiments/all_coco_prompts.txt"

mkdir -p "${RESULTS_DIR}"

CUDA_VISIBLE_DEVICES=0 python ./src/steering/apply_steering_with_injection_flux_1.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "${DATA_DIR}" \
    --task nudity \
    --strength 45.0 \
    --strength_txt 6.0 \
    --strength_img 0.0 \
    --top_k_percent 0.95 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --injection_point block \
    --results_dir "${RESULTS_DIR}" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type diff \
    --steering_type separate \
    --guidance_scale 0.0 \
    --width 512 \
    --height 512 \
    --use_cls \
    --steer_txt \
    --prompts_path "${PROMPTS_PATH}"
