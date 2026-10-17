#!/bin/bash
# Steps 1-2 for the seven I2P categories (135 pairs each, same base prompts as nudity).
# Produces experiments/FLUX.1-schnell/<category>/steering, referenced by
# configs/multi_concept_i2p.json.
set -euo pipefail

for CONCEPT in "hate" "harassment" "violence" "self-harm" "sexual" "shocking" "illegal content"; do
    CONCEPT="${CONCEPT}" TASK=people NUM_PROMPTS=135 bash scripts/get_vector.sh
    CONCEPT="${CONCEPT}" NUM_PROMPTS=135 bash scripts/steering_calculate.sh
done
