#!/bin/bash
# Step 3/3 (FLUX): concept addition / switching (e.g. add a smile, woman -> man).
# The same mean-difference field is applied with a positive sign; no classifier.
# Build the vector first, e.g. for "smile":
#   CONCEPT=smile TASK=people NUM_PROMPTS=20 bash scripts/get_vector.sh
#   CONCEPT=smile NUM_PROMPTS=20 bash scripts/steering_calculate.sh
#
#   CONCEPT=smile STRENGTH=20 bash scripts/add_concept_flux.sh
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
GS="${GS:-0.0}"
STEPS="${STEPS:-4}"
CONCEPT="${CONCEPT:-smile}"
STRENGTH="${STRENGTH:-20}"
PROMPTS_PATH="${PROMPTS_PATH:-prompts_collection/ablation/ablation_prompts_add.txt}"
VECTOR_DIR="${VECTOR_DIR:-experiments/FLUX.1-schnell/${CONCEPT// /_}/steering}"
RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/add_s${STRENGTH}}"

python ./src/steering/apply_steering_with_injection_flux.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${VECTOR_DIR}" \
    --task "add concept" \
    --prompts_path "${PROMPTS_PATH}" \
    --injection_point block \
    --vector_type diff \
    --steering_type separate \
    --strength "${STRENGTH}" \
    --strength_img 0.0 \
    --inference_steps "${STEPS}" \
    --guidance_scale "${GS}" \
    --width 1024 \
    --height 1024 \
    --results_dir "${RESULTS_DIR}"
