#!/bin/bash
set -euo pipefail

# Subspace-AcT inference steering.
# Run from repo root:
#   bash scripts/steering_apply_subspace_act.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
# Must be the directory that contains base_*_*_txt-*_subspace_act_v1.pt from calculate (match SAVE_DIR there).
DATA_DIR="${DATA_DIR:-/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_add/final_steering/subspace_act_txt_meandiff_tokenwise_raw_delta_attn_tokenwise_T_top_meandiff_100_100_main_centered}"
PROMPTS_PATH="${PROMPTS_PATH:-data/captions.txt}"
RESULTS_DIR="${RESULTS_DIR:-/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_add/generated_images/subspace/attn/mean_diff_top_affine_100_100_main_centered}"
  #/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_add/generated_images/subspace/attn/mean_diff_top_affine_100_1_main
VECTOR_TYPE="${VECTOR_TYPE:-subspace_act_v1}"
INJECTION_POINT="${INJECTION_POINT:-attn}"   # attn | block | ff
TASK="${TASK:-add concept}"                   # add concept | remove | nudity

WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"
SEED="${SEED:-42}"

STRENGTH_TXT="${STRENGTH_TXT:-0.0}"          # alpha for txt branch
# Text-only default: keep image branch disabled.
STRENGTH_IMG="${STRENGTH_IMG:-0.0}"           # alpha for img branch
LAMBDA_TXT="${LAMBDA_TXT:-1.0}"               # lambda for txt transport blend
LAMBDA_IMG="${LAMBDA_IMG:-0.0}"               # lambda for img transport blend

BLOCK_STEERING="${BLOCK_STEERING:-all}"
T_STEERING="${T_STEERING:-all}"

STRUCTURE="${STRUCTURE:-0.5}"
ENERGY_RESTORATION="${ENERGY_RESTORATION:-0}"

# Baseline dump: путь пустой = выключено. Нужны STRENGTH_TXT/STRENGTH_IMG = 0 (см. выше).
DUMP_BASELINE_PROMPT_IDX="${DUMP_BASELINE_PROMPT_IDX:-0}"

# subspace_meandiff only: strip ΔA along top-k PCA of current activations (0 = off).
MEANDIFF_NEUTRAL_K="${MEANDIFF_NEUTRAL_K:-0}"

echo "Running Subspace-AcT inference..."
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  DATA_DIR=${DATA_DIR}"
echo "  RESULTS_DIR=${RESULTS_DIR}"
echo "  ENERGY_RESTORATION=${ENERGY_RESTORATION}"


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
  --prompts_path "/home/jovyan/konovalova/clean_code/steering/prompts_collection/ablation/ablation_prompts_add.txt" \
  --energy_restoration \
  --meandiff_neutral_k "${MEANDIFF_NEUTRAL_K}" \
  --dump_baseline_activations "./activation_archive_prompt0.pt" \
  --dump_baseline_prompt_idx "${DUMP_BASELINE_PROMPT_IDX}"

echo "Done."
