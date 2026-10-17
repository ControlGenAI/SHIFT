#!/bin/bash
# Step 2/3 (FLUX): steering field, linear SVMs and pooled text vector from the
# dumps of scripts/get_vector.sh. Use the same CONCEPT / NUM_PROMPTS / MODEL_NAME.
#
#   bash scripts/steering_calculate.sh
#   CONCEPT=Snoopy NUM_PROMPTS=20 bash scripts/steering_calculate.sh
#
# Writes to ${EXP_DIR}/steering:
#   base_0.85_<N>_diff.pt        per-block, per-token mean difference (DiT route)
#   base_0.85_<N>_svm_models.pt  per-block linear SVMs + _scores.pt / _normals.pt (--use_cls)
#   base_0.85_<N>_text_diff.pt   pooled / sequence text-encoder difference (--steer_txt, gate)
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
CONCEPT="${CONCEPT:-nudity}"
NUM_PROMPTS="${NUM_PROMPTS:-135}"
EXP_DIR="${EXP_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}}"
SAVE_DIR="${EXP_DIR}/steering"

POS_PATH=$(ls "${EXP_DIR}"/dumps/*_prompts_"${NUM_PROMPTS}"_pos_block.pt)
NEG_PATH=$(ls "${EXP_DIR}"/dumps/*_prompts_"${NUM_PROMPTS}"_neg_block.pt)
POS_TXT_PATH=$(ls "${EXP_DIR}"/dumps_txt/*_prompts_"${NUM_PROMPTS}"_pos_embeddings.pt)
NEG_TXT_PATH=$(ls "${EXP_DIR}"/dumps_txt/*_prompts_"${NUM_PROMPTS}"_neg_embeddings.pt)

mkdir -p "${SAVE_DIR}"

# Linear SVMs (classifier scale eta_cls, --use_cls)
python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method svm \
    --save_svm \
    --timesteps 4 \
    --blocks 19 \
    --n_samples "${NUM_PROMPTS}" \
    --classifier none

# Mean-difference steering field
python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method diff \
    --timesteps 4 \
    --blocks 19 \
    --n_samples "${NUM_PROMPTS}" \
    --classifier none

# Pooled text-encoder vector
python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_TXT_PATH}" \
    --neg_path "${NEG_TXT_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method text \
    --n_samples "${NUM_PROMPTS}"
