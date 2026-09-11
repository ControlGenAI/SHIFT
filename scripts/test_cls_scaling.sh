#!/usr/bin/env bash
# Run on the GPU cluster from the SHIFT repository root, with its venv active.
set -euo pipefail

if [[ $# -gt 4 ]]; then
  echo 'Usage: bash scripts/test_cls_scaling.sh [dataset] [direction.pt] [new_output_dir] [num_samples]' >&2
  exit 2
fi
cls_dataset="${1:-out/dataset200}"
cls_direction="${2:-out/cls_features/cls_direction.pt}"
cls_output="${3:-out/cls_scaling_check}"
cls_count="${4:-1}"

if [[ ! -f "$cls_dataset/dataset.json" || ! -f "$cls_direction" ]]; then
  echo 'Dataset manifest or CLS direction is missing; pass the existing cluster paths.' >&2
  exit 2
fi
if [[ -e "$cls_output" ]]; then
  echo "Output already exists: $cls_output. Choose a new directory." >&2
  exit 2
fi
if [[ ! "$cls_count" =~ ^[1-9][0-9]*$ ]]; then
  echo 'num_samples must be a positive integer.' >&2
  exit 2
fi
for cls_variant in rms none_matched none; do
  if [[ ! -f "configs/cls_joint_scaling_${cls_variant}.json" ]]; then
    echo 'Run this script from the SHIFT repository root.' >&2
    exit 2
  fi
done

for cls_variant in rms none_matched none; do
  python -m src.dino_adapter.cls_experiment \
    --config "configs/cls_joint_scaling_${cls_variant}.json" \
    optimize --dataset "$cls_dataset" --split test --num-samples "$cls_count" \
    --direction "$cls_direction" --output "$cls_output/$cls_variant"
done
