#!/bin/bash
# Optional: compute text-embedding steering vectors (FLUX / SD3.5).
# Requires dumps from get_encoding_vector.py.
# Run from repository root: bash scripts/steering_calculate_txt.sh
set -euo pipefail

POS_TXT_PATH="experiments/flux_schnell/remove/data_vectors_txt/people_nudity_nudity_prompts_135_pos_embeddings.pt"
NEG_TXT_PATH="experiments/flux_schnell/remove/data_vectors_txt/people__nudity_prompts_135_neg_embeddings.pt"
SAVE_DIR="experiments/flux_schnell/remove/final_steering/block_steering/nudity"

mkdir -p "${SAVE_DIR}"

python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_TXT_PATH}" \
    --neg_path "${NEG_TXT_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method text \
    --n_samples 135
