#!/bin/bash
# Step 3/3 (FLUX): erase the seven I2P categories jointly. Every concept keeps its
# own steering field, SVM scale and cosine gate (gamma=225, delta=0.15).
# Run scripts/prepare_multi_concept.sh first. Q16 evaluation: see metrics/README.md.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
GS="${GS:-0.0}"
STEPS="${STEPS:-4}"
VECTOR_ROOT="${VECTOR_ROOT:-experiments/FLUX.1-schnell}"
CONFIG="${CONFIG:-configs/multi_concept_i2p.json}"
RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/multi_concept/i2p}"

python ./src/steering/apply_steering_with_injection_flux_multi.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${VECTOR_ROOT}" \
    --steering_config "${CONFIG}" \
    --task nudity \
    --injection_point block \
    --vector_type diff \
    --steering_type separate \
    --use_cls \
    --cls_min 20.0 \
    --min_signal_threshold 0.05 \
    --steer_txt \
    --use_pooled_cosine_score \
    --pooled_cosine_clip "${DELTA:-0.15}" \
    --inference_steps "${STEPS}" \
    --guidance_scale "${GS}" \
    --width 512 \
    --height 512 \
    --results_dir "${RESULTS_DIR}"
