#!/bin/bash
set -euo pipefail

# Subspace-AcT + явный текстовый вектор (classic *_text_diff.pt, Monge/subspace из stats .pt).
# Запуск из корня репозитория steering:
#   bash scripts/steering_apply_subspace_act_txt_pooled.sh
#
# Примеры:
#   Авто {prefix}_text_diff.pt в DATA_DIR (как раньше, но STRENGTH_TXT из env):
#     bash scripts/steering_apply_subspace_act_txt_pooled.sh
#
#   Monge из calculate_text_encoder_pooled_monge.py:
#     TEXT_VECTOR_PATH=.../text_pooled_monge_stats.pt TEXT_VECTOR_KEY=txt_steering_vector_monge \
#     RESULTS_DIR=.../out_monge bash scripts/steering_apply_subspace_act_txt_pooled.sh
#
#   Принудительно классический steering_txt_data:
#     LEGACY_TXT_STEERING=1 TEXT_VECTOR_PATH=.../prefix_text_diff.pt bash ...

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
DATA_DIR="${DATA_DIR:-/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/final_steering/subspace_act_txt_meandiff_tokenwise_raw_delta_attn_tokenwise_T_top_meandiff_100_nudity}"
PROMPTS_PATH="${PROMPTS_PATH:-/home/jovyan/konovalova/steering/all_coco_prompts.txt}"
RESULTS_DIR="${RESULTS_DIR:-/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/generated_images_big/subspace/attn/test}"

VECTOR_TYPE="${VECTOR_TYPE:-subspace_act_v1}"
INJECTION_POINT="${INJECTION_POINT:-attn}"
TASK="${TASK:-nudity}"

WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"
SEED="${SEED:-42}"

STRENGTH_TXT="${STRENGTH_TXT:-10.0}"
STRENGTH_IMG="${STRENGTH_IMG:-0.0}"
LAMBDA_TXT="${LAMBDA_TXT:-1.0}"
LAMBDA_IMG="${LAMBDA_IMG:-0.0}"

BLOCK_STEERING="${BLOCK_STEERING:-all}"
T_STEERING="${T_STEERING:-all}"
STRUCTURE="${STRUCTURE:-0.5}"
MEANDIFF_NEUTRAL_K="${MEANDIFF_NEUTRAL_K:-0}"

TEXT_VECTOR_PATH="${TEXT_VECTOR_PATH:-}"
TEXT_VECTOR_KEY="${TEXT_VECTOR_KEY:-}"
LEGACY_TXT_STEERING="${LEGACY_TXT_STEERING:-0}"

EXTRA_TXT_ARGS=()
if [[ -n "${TEXT_VECTOR_PATH}" ]]; then
  EXTRA_TXT_ARGS+=(--text_vector_path "${TEXT_VECTOR_PATH}")
fi
if [[ -n "${TEXT_VECTOR_KEY}" ]]; then
  EXTRA_TXT_ARGS+=(--text_vector_key "${TEXT_VECTOR_KEY}")
fi
if [[ "${LEGACY_TXT_STEERING}" == "1" ]]; then
  EXTRA_TXT_ARGS+=(--legacy_txt_steering)
fi

echo "Subspace-AcT + txt (pooled / explicit path)"
echo "  DATA_DIR=${DATA_DIR}  RESULTS_DIR=${RESULTS_DIR}"
echo "  STRENGTH_TXT=${STRENGTH_TXT}  STRENGTH_IMG=${STRENGTH_IMG}"
if [[ -n "${TEXT_VECTOR_PATH}" ]]; then
  echo "  TEXT_VECTOR_PATH=${TEXT_VECTOR_PATH}  TEXT_VECTOR_KEY=${TEXT_VECTOR_KEY:-<whole file>}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python ./src/steering/apply_steering_with_injection_flux_subspace_act.py \
  --model_name "${MODEL_NAME}" \
  --data_dir "${DATA_DIR}" \
  --vector_type "${VECTOR_TYPE}" \
  --task "${TASK}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --injection_point "${INJECTION_POINT}" \
  --strength "${STRENGTH_TXT}" \
  --strength_img "${STRENGTH_IMG}" \
  --transport_lambda_txt "${LAMBDA_TXT}" \
  --transport_lambda_img "${LAMBDA_IMG}" \
  --block_steering "${BLOCK_STEERING}" \
  --t_steering "${T_STEERING}" \
  --inference_steps "${INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --seed "${SEED}" \
  --structure "${STRUCTURE}" \
  --results_dir "${RESULTS_DIR}" \
  --prompts_path "${PROMPTS_PATH}" \
  --meandiff_neutral_k "${MEANDIFF_NEUTRAL_K}" \
  --energy_restoration \
  --steer_txt \
  --strength_txt "6" \
  --use_cls

echo "Done."
