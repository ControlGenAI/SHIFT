#!/bin/bash
# Step 1/3 (FLUX): record post-block text activations and text-encoder embeddings
# for contrastive prompt pairs (p-, p+). Run from the repository root:
#
#   bash scripts/get_vector.sh                                            # nudity, 135 pairs
#   CONCEPT=Snoopy TASK=concrete NUM_PROMPTS=20 \
#     PROMPT_PATH=prompts_collection/dataset_creation/dataset_prompts_remove_2.txt bash scripts/get_vector.sh
#   CONCEPT="Van Gogh" TASK=style NUM_PROMPTS=20 \
#     PROMPT_PATH=prompts_collection/dataset_creation/dataset_prompts_style.txt bash scripts/get_vector.sh
#
# TASK sets the pair template: people "{base}, {concept}" (the 15x9 base prompts are
# built in code), concrete "{base} with {concept}", style "{base}, {concept} style",
# switch "{concept} {base}". FLUX.1-dev: MODEL_NAME=black-forest-labs/FLUX.1-dev GS=3.5 STEPS=28.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
CONCEPT="${CONCEPT:-nudity}"
TASK="${TASK:-people}"
NUM_PROMPTS="${NUM_PROMPTS:-135}"
PROMPT_PATH="${PROMPT_PATH:-prompts_collection/dataset_creation/dataset_prompts_remove_2.txt}"
GS="${GS:-0.0}"
STEPS="${STEPS:-4}"
EXP_DIR="${EXP_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}}"

mkdir -p "${EXP_DIR}/dumps" "${EXP_DIR}/dumps_txt" "${EXP_DIR}/dataset_images"

# --- DiT activations: post-block text stream of the 19 double-stream blocks ---
# Only the step-0 field is used at inference; later steps are kept for the
# per-step ablation (--per_step_vectors).
python ./src/steering/get_vector_1.py \
    --model_name "${MODEL_NAME}" \
    --task "${TASK}" \
    --pos_concept "${CONCEPT}" \
    --neg_concept "" \
    --num_prompts "${NUM_PROMPTS}" \
    --extraction_point block \
    --token_stream txt \
    --num_layers 19 \
    --save_timesteps 4 \
    --height 1024 \
    --width 1024 \
    --gs "${GS}" \
    --num_inference_steps "${STEPS}" \
    --batch_size 1 \
    --exp_type "${CONCEPT// /_}" \
    --save_dir "${EXP_DIR}/dumps" \
    --save_image_dir "${EXP_DIR}/dataset_images" \
    --prompt_path "${PROMPT_PATH}"

# --- Text-encoder embeddings (pooled CLIP route and cosine gate) ---
python ./src/steering/get_encoding_vector.py \
    --model_name "${MODEL_NAME}" \
    --task "${TASK}" \
    --pos_concept "${CONCEPT}" \
    --neg_concept "" \
    --num_prompts "${NUM_PROMPTS}" \
    --save_dir "${EXP_DIR}/dumps_txt" \
    --prompt_path "${PROMPT_PATH}"
