#!/bin/bash
set -euo pipefail

# Subspace-AcT vector calculation — defaults: text stream + subspace_meandiff (no OT/Sinkhorn fit on z).
# Run from repo root:
#   bash scripts/steering_calculate_subspace_act.sh
#
# Affine / Sinkhorn on z:  TXT_MODE=subspace_affine  or  TXT_MODE=subspace_sinkhorn
# Image stream or full dual:  BRANCH=img  or  BRANCH=dual
#
# PCA on paired deltas: PCA_DELTA_MODE=centered | raw_delta | append_mean_dir

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# NOTE: This script consumes activation *.pt files (from get_vector_1.py),
# not rendered images from dataset_images_attn.
NEG_PATH="${NEG_PATH:-/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/data_vector/data_vector_attn_100/nudity__gs_0.0_prompts_150_neg_block.pt}"
POS_PATH="${POS_PATH:-/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/data_vector/data_vector_attn_100/nudity__gs_0.0_prompts_150_pos_block.pt}"
SAVE_DIR="${SAVE_DIR:-experiments/flux_schnell/test_compare_remove/final_steering/subspace_act_txt_meandiff_tokenwise_raw_delta_block_tokenwise_T_top_meandiff_100_nudity_block_neutral_k_1}"

# branch: txt (default) | img | dual
BRANCH="${BRANCH:-txt}"

# Modes: subspace_affine | subspace_sinkhorn | subspace_meandiff (mean paired shift in z, no transport fit)
IMG_MODE="${IMG_MODE:-subspace_affine}"
TXT_MODE="${TXT_MODE:-subspace_meandiff}"

TIMESTEPS="${TIMESTEPS:-4}"
BLOCKS="${BLOCKS:-19}"
N_SAMPLES="${N_SAMPLES:-20}"
SUBSPACE_DIM="${SUBSPACE_DIM:-32}"
MIN_EXPLAINED="${MIN_EXPLAINED:-0.90}"
PCA_DELTA_MODE="${PCA_DELTA_MODE:-raw_delta}"
EPSILON="${EPSILON:-0.05}"
SINKHORN_ITER="${SINKHORN_ITER:-100}"
THRESHOLD="${THRESHOLD:-0.85}"
# activation | dual | text — OT2-compatible; text-only activations use activation (default).
# Do NOT set this to "txt"/"img" (those are --branch values).
DATA_TYPE="${DATA_TYPE:-activation}"
if [[ "${DATA_TYPE}" == "txt" || "${DATA_TYPE}" == "img" ]]; then
  echo "WARNING: DATA_TYPE='${DATA_TYPE}' is invalid (use BRANCH=${DATA_TYPE} for which stream to fit). Using DATA_TYPE=dual."
  DATA_TYPE=dual
fi

echo "Running Subspace-AcT vector calculation..."
echo "  NEG_PATH=${NEG_PATH}"
echo "  POS_PATH=${POS_PATH}"
echo "  SAVE_DIR=${SAVE_DIR}"
echo "  BRANCH=${BRANCH} DATA_TYPE=${DATA_TYPE}"
echo "  IMG_MODE=${IMG_MODE} TXT_MODE=${TXT_MODE}"
echo "  PCA_DELTA_MODE=${PCA_DELTA_MODE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python ./src/steering/calculate_steering_vector_subspace_act.py \
  --neg_path "${NEG_PATH}" \
  --pos_path "${POS_PATH}" \
  --branch "${BRANCH}" \
  --method sinkhorn \
  --img_mode "${IMG_MODE}" \
  --txt_mode "${TXT_MODE}" \
  --timesteps "${TIMESTEPS}" \
  --blocks "${BLOCKS}" \
  --n_samples "${N_SAMPLES}" \
  --subspace_dim "${SUBSPACE_DIM}" \
  --min_explained "${MIN_EXPLAINED}" \
  --pca_delta_mode "${PCA_DELTA_MODE}" \
  --epsilon "${EPSILON}" \
  --sinkhorn_iter "${SINKHORN_ITER}" \
  --threshold "${THRESHOLD}" \
  --data_type "${DATA_TYPE}" \
  --save_dir "${SAVE_DIR}" \
  --cpca_tokenwise \
  --meandiff_neutral_k 1 \

echo "Done."
