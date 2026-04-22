# #!/bin/bash
# # Step 1: Extract attention activations (get_vector.py)
# # Step 2: Extract text embeddings (get_encoding_vector.py)
# # Run from repository root: bash scripts/get_vector.sh

# NUM_PROMPTS=135
# POS_CONCEPT_PROMPT='Nudity'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="nudity_135"

# SAVE_IMAGE_DIR="experiments/flux_schnell/remove/dataset_images"
# SAVE_BASE_DIR="experiments/flux_schnell/remove/data_vectors"
# SAVE_TXT_DIR="experiments/flux_schnell/remove/data_vectors_txt"

# mkdir -p "$SAVE_BASE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"
# mkdir -p "$SAVE_TXT_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Step 1: Extract attention activations ---
# CUDA_VISIBLE_DEVICES=0 python ./src/steering/get_vector.py \
#     --task "people" \
#     --num_prompts "$NUM_PROMPTS" \
#     --pos_concept "$POS_CONCEPT_PROMPT" \
#     --neg_concept "$NEG_CONCEPT_PROMPT" \
#     --save_dir "$SAVE_BASE_DIR" \
#     --save_image_dir "$SAVE_IMAGE_DIR" \
#     --exp_type "$POS_CONCEPT_KEY" \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --gs 0.0 \
#     --num_inference_steps 4 \
#     --batch_size 1 \
#     --prompt_path "prompts_collection/dataset_creation/dataset_prompts_remove_3.txt"

# echo "Done. Activations saved in $SAVE_BASE_DIR"

# # --- Step 2: Extract text embeddings ---
# python ./src/steering/get_encoding_vector.py \
#     --pos_concept "$POS_CONCEPT_PROMPT" \
#     --neg_concept "$NEG_CONCEPT_PROMPT" \
#     --num_prompts "$NUM_PROMPTS" \
#     --task "people" \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --save_dir "$SAVE_TXT_DIR" \
#     --prompt_path "prompts_collection/dataset_creation/dataset_prompts_remove_3.txt"

# echo "Done. Text embeddings saved in $SAVE_TXT_DIR"

# Block-level extraction (residual stream) — recommended
# CUDA_VISIBLE_DEVICES=0 python ./src/steering/get_vector_1.py \
#     --height 1024 \
#     --width 1024 \
#     --model_name black-forest-labs/FLUX.1-dev \
#     --task concrete --pos_concept glasses \
#     --exp_type glasses_20 \
#     --num_prompts 20 --num_layers 19 --save_timesteps 4 \
#     --extraction_point block \
#     --save_image_dir experiments/flux_schnell/add/dataset_images \
#     --save_dir experiments/flux_schnell/add/data_vectors \
#     --prompt_path "prompts_collection/dataset_creation/dataset_prompts_add.txt"

# python ./src/steering/calculate_steering_vector_ot_2.py \
#     --pos_path '/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vectors/glasses_20_realistic_gs_4.5_prompts_20_neg_block.pt' \
#     --neg_path '/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vectors/glasses_20_realistic_gs_4.5_prompts_20_neg_block.pt' \
#     --save_dir /home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/final_steering/glasses_new \
#     --data_type dual \
#     --method sinkhorn \
#     --max_tokens_sinkhorn 1024



python ./src/steering/get_vector_1.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --task "concrete" \
    --pos_concept "glasses" \
    --neg_concept "" \
    --num_prompts 100 \
    --extraction_point "attn" \
    --num_layers "19" \
    --save_timesteps "4" \
    --height "1024" \
    --width "1024" \
    --gs "0.0" \
    --num_inference_steps "4" \
    --batch_size 1 \
    --save_dir "experiments/flux_schnell/test_compare_add/data_vector/data_vector_attn_100" \
    --exp_type "glasses" \
    --token_stream txt \
    --prompt_path "prompts_collection/dataset_creation/dataset_prompts_add.txt" \
    --save_image_dir "experiments/flux_schnell/test_compare_add/dataset_images/dataset_images_attn_100"