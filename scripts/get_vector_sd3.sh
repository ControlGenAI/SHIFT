#!/bin/bash
# Step 1/3 (SD3.5): extract block activations + text embeddings for a concept.
# Run from repository root: bash scripts/get_vector_sd3.sh
set -euo pipefail

MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
EXP_TYPE="nudity"
POS_CONCEPT="nudity"
NUM_PROMPTS=135
SAVE_DIR="experiments/sd35/remove/data_vectors"
SAVE_TXT_DIR="experiments/sd35/remove/data_vectors_txt"
SAVE_IMAGE_DIR="experiments/sd35/remove/dataset_images"
PROMPT_PATH="prompts_collection/dataset_creation/dataset_prompts_remove_3.txt"

mkdir -p "${SAVE_DIR}" "${SAVE_TXT_DIR}" "${SAVE_IMAGE_DIR}"

# --- Activations ---
CUDA_VISIBLE_DEVICES=0 python ./src/steering/get_vector_sd35.py \
    --model_name "${MODEL_NAME}" \
    --task people \
    --pos_concept "${POS_CONCEPT}" \
    --neg_concept "" \
    --num_prompts "${NUM_PROMPTS}" \
    --extraction_point block \
    --token_stream both \
    --num_layers 24 \
    --save_timesteps 4 \
    --height 1024 \
    --width 1024 \
    --gs 4.5 \
    --num_inference_steps 28 \
    --batch_size 1 \
    --exp_type "${EXP_TYPE}" \
    --save_dir "${SAVE_DIR}" \
    --save_image_dir "${SAVE_IMAGE_DIR}" \
    --prompt_path "${PROMPT_PATH}"

# --- Text embeddings (for --steer_txt / pooled cosine) ---
python ./src/steering/get_encoding_vector.py \
    --model_name "${MODEL_NAME}" \
    --task people \
    --pos_concept "${POS_CONCEPT}" \
    --neg_concept "" \
    --num_prompts "${NUM_PROMPTS}" \
    --save_dir "${SAVE_TXT_DIR}" \
    --prompt_path "${PROMPT_PATH}"
