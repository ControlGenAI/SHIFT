#!/bin/bash
# Quality preservation (FID / CLIP on MS-COCO) for a removal vector, 512x512.
# Captions and per-caption seeds come from prompts_collection/; NUM_PROMPTS limits
# the run (the paper reports MS-COCO 5K). Same GATE / STRENGTH / MODEL_NAME switches
# as scripts/erase_nudity_flux.sh.
#
#   bash scripts/eval_coco_flux.sh
#   python metrics/clip_coco.py --image_dir ${RESULTS_DIR}/steered
#   bash metrics/fid.sh <coco_real_dir> ${RESULTS_DIR}/steered
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
GS="${GS:-0.0}"
STEPS="${STEPS:-4}"
GATE="${GATE:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-5000}"
VECTOR_DIR="${VECTOR_DIR:-experiments/FLUX.1-schnell/nudity/steering}"

if [[ "${GATE}" == "1" ]]; then
    STRENGTH="${STRENGTH:-225}"
    GATE_ARGS=(--use_pooled_cosine_score --pooled_cosine_clip "${DELTA:-0.15}")
else
    STRENGTH="${STRENGTH:-30}"
    GATE_ARGS=()
fi
RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/nudity/coco_s${STRENGTH}_gate${GATE}}"

python ./src/steering/apply_steering_with_injection_flux.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${VECTOR_DIR}" \
    --task remove \
    --prompts_path prompts_collection/main_experiments/all_coco_prompts.txt \
    --seeds_path prompts_collection/additiona_seeds/coco_seeds.txt \
    --num_prompts "${NUM_PROMPTS}" \
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
