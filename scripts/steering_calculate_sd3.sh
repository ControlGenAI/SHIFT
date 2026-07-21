#!/bin/bash
# Step 2/3 (SD3.5): compute steering vectors from extracted activations / embeddings.
# Edit paths to match files produced by scripts/get_vector_sd3.sh.
# Run from repository root: bash scripts/steering_calculate_sd3.sh
set -euo pipefail

# Activation dumps (empty neg_concept -> double underscore in the filename)
POS_PATH="experiments/sd35/remove/data_vectors/nudity__gs_4.5_prompts_135_pos_block.pt"
NEG_PATH="experiments/sd35/remove/data_vectors/nudity__gs_4.5_prompts_135_neg_block.pt"

# Text-embedding dumps from get_encoding_vector.py — adjust names if needed
POS_TXT_PATH="experiments/sd35/remove/data_vectors_txt/people_nudity_nudity_prompts_135_pos_embeddings.pt"
NEG_TXT_PATH="experiments/sd35/remove/data_vectors_txt/people__nudity_prompts_135_neg_embeddings.pt"

SAVE_DIR="experiments/sd35/remove/final_steering/block_steering/nudity"

mkdir -p "${SAVE_DIR}"

# Classifier + scores (needed for --use_cls at apply time)
python ./src/steering/calclulate_steering_vectors_sd35.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method svm \
    --save_svm \
    --timesteps 4 \
    --blocks 24 \
    --n_samples 135 \
    --token_stream both \
    --classifier none \
    --threshold 0.85

# Mean-difference activation steering vector
python ./src/steering/calclulate_steering_vectors_sd35.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method diff \
    --timesteps 4 \
    --blocks 24 \
    --n_samples 135 \
    --token_stream both \
    --classifier none \
    --threshold 0.85

# Text-embedding steering vector (for --steer_txt)
if [[ -f "${POS_TXT_PATH}" && -f "${NEG_TXT_PATH}" ]]; then
  python ./src/steering/calclulate_steering_vectors_sd35.py \
      --pos_path "${POS_TXT_PATH}" \
      --neg_path "${NEG_TXT_PATH}" \
      --save_dir "${SAVE_DIR}" \
      --method text \
      --n_samples 135 \
      --token_stream both \
      --threshold 0.85
else
  echo "WARNING: text embedding files not found; skipping text vector."
  echo "  expected: ${POS_TXT_PATH}"
  echo "  expected: ${NEG_TXT_PATH}"
fi
