#!/bin/bash
# Step 3/3 (FLUX): concrete-concept erasure, SPM protocol (80 prompt templates,
# target + preserved concepts). One run per evaluated concept and seed; the
# concept name is appended to every template (--prompt_suffix).
#
#   bash scripts/erase_concept_flux.sh                         # erase Snoopy, gate off, gamma=20
#   GATE=1 bash scripts/erase_concept_flux.sh                  # cosine gate, gamma=225
#   SEEDS="1 2 3" bash scripts/erase_concept_flux.sh           # the paper averages over 9 seeds
#
# Evaluate: python metrics/clip_erase.py --target_dirs ${RESULTS_ROOT}/seed_*/steered
#           python metrics/fid_erase.py --origin_dirs <unsteered seed dirs> --target_dirs ...
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
GS="${GS:-0.0}"
STEPS="${STEPS:-4}"
GATE="${GATE:-0}"
CONCEPT="${CONCEPT:-Snoopy}"
EVAL_CONCEPTS="${EVAL_CONCEPTS:-Snoopy Mickey Spongebob Pikachu Dog Legislator}"
SEEDS="${SEEDS:-42}"
PROMPTS_PATH="${PROMPTS_PATH:-prompts_collection/main_experiments/ablation_prompts_remove.txt}"
VECTOR_DIR="${VECTOR_DIR:-experiments/FLUX.1-schnell/${CONCEPT// /_}/steering}"

if [[ "${GATE}" == "1" ]]; then
    STRENGTH="${STRENGTH:-225}"
    GATE_ARGS=(--use_pooled_cosine_score --pooled_cosine_clip "${DELTA:-0.15}")
else
    STRENGTH="${STRENGTH:-20}"
    GATE_ARGS=()
fi
RESULTS_ROOT="${RESULTS_ROOT:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/spm_s${STRENGTH}_gate${GATE}}"

for SEED in ${SEEDS}; do
  for EVAL_CONCEPT in ${EVAL_CONCEPTS}; do
    python ./src/steering/apply_steering_with_injection_flux.py \
        --model_name "${MODEL_NAME}" \
        --data_dir "${VECTOR_DIR}" \
        --task remove \
        --prompts_path "${PROMPTS_PATH}" \
        --prompt_suffix "${EVAL_CONCEPT}" \
        --seed "${SEED}" \
        --injection_point block \
        --vector_type diff \
        --steering_type separate \
        --strength "${STRENGTH}" \
        --strength_img 0.0 \
        --use_cls \
        --cls_min 20.0 \
        --min_signal_threshold 0.05 \
        --steer_txt \
        --strength_txt 6.0 \
        "${GATE_ARGS[@]+"${GATE_ARGS[@]}"}" \
        --inference_steps "${STEPS}" \
        --guidance_scale "${GS}" \
        --width 1024 \
        --height 1024 \
        --results_dir "${RESULTS_ROOT}/seed_${SEED}"
  done
done
