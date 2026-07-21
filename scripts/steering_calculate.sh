#!/bin/bash
# Step 2/3 (FLUX): compute steering vectors from extracted activations.
# Edit paths to match files from scripts/get_vector.sh.
# Run from repository root: bash scripts/steering_calculate.sh
set -euo pipefail

POS_PATH="experiments/flux_schnell/remove/data_vectors/nudity__gs_0.0_prompts_135_pos_block.pt"
NEG_PATH="experiments/flux_schnell/remove/data_vectors/nudity__gs_0.0_prompts_135_neg_block.pt"
SAVE_DIR="experiments/flux_schnell/remove/final_steering/block_steering/nudity"

mkdir -p "${SAVE_DIR}"

# Classifier / scores (for --use_cls)
python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --save_svm \
    --timesteps 4 \
    --blocks 19 \
    --n_samples 135 \
    --classifier none

# Mean-difference steering vector
python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method diff \
    --n_samples 135 \
    --classifier none
