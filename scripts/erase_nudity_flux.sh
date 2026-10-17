#!/bin/bash
# Step 3/3 (FLUX): nudity erasure on I2P (4,703 prompts with their seeds), 512x512.
#
#   bash scripts/erase_nudity_flux.sh                    # FLUX.1-schnell, gate off, gamma=30
#   GATE=1 bash scripts/erase_nudity_flux.sh             # shared cosine gate, gamma=225, delta=0.15
#   MODEL_NAME=black-forest-labs/FLUX.1-dev GS=3.5 STEPS=28 bash scripts/erase_nudity_flux.sh
#
# VECTOR_DIR defaults to the FLUX.1-schnell vectors, so the FLUX.1-dev run is the
# cross-distillation transfer ("schnell acts"); point it at dev dumps for "dev acts".
# Evaluate with: python metrics/nudenet_eval.py --image_dir ${RESULTS_DIR}/steered
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
GS="${GS:-0.0}"
STEPS="${STEPS:-4}"
GATE="${GATE:-0}"
VECTOR_DIR="${VECTOR_DIR:-experiments/FLUX.1-schnell/nudity/steering}"

if [[ "${GATE}" == "1" ]]; then
    STRENGTH="${STRENGTH:-225}"
    GATE_ARGS=(--use_pooled_cosine_score --pooled_cosine_clip "${DELTA:-0.15}")
else
    STRENGTH="${STRENGTH:-30}"
    GATE_ARGS=()
fi
RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/nudity/i2p_s${STRENGTH}_gate${GATE}}"

python ./src/steering/apply_steering_with_injection_flux.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${VECTOR_DIR}" \
    --task nudity \
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
    --width 512 \
    --height 512 \
    --results_dir "${RESULTS_DIR}"
