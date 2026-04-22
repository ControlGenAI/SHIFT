#!/bin/bash
# Calculate OT-based steering vectors from extracted activations and text embeddings.
# Run from repository root: bash scripts/steering_calculate_ot.sh
#
# Outputs (under --save_dir):
#   base_{threshold}_{n_samples}_ot_sw.pt   or _ot_sinkhorn.pt
#   base_{threshold}_{n_samples}_text_ot_sw.pt  (when --data_type text)

# --- Example: activation OT (Sliced Wasserstein) ---
# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#     --method sw \
#     --n_projections 200 \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sw"

# # --- Example: activation OT (Sinkhorn) ---
# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#     --method sinkhorn \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin"

# --- Example: text-embedding OT ---
# python ./src/steering/calculate_steering_vectors_ot.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
#     --method sw \
#     --data_type text \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/remove/final_steering/test"

# python ./src/steering/calculate_steering_vectors_ot.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
#     --method sinkhorn \
#     --data_type text \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin"

# python ./src/steering/calculate_steering_vectors.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
#     --method  \
#     --data_type text \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin"

# python ./src/steering/calculate_steering_vectors.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
#     --method text \
#     --n_samples 20 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_diff"


# python ./src/steering/calculate_steering_vectors.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#     --method 'diff' \
#     --n_samples 20 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_diff"


# --- Example: activation OT (Sinkhorn) ---
# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot1.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#     --method sinkhorn \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_new"

# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot1.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#     --method sinkhorn \
#     --use_partial_ot \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_new_partial_ot"

# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot1.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#     --method sw \
#     --subspace_dim 64 \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sw_new"



# python ./src/steering/calculate_steering_vectors_ot1.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
#     --method sinkhorn \
#     --data_type text \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_new"


# python ./src/steering/calculate_steering_vectors_ot1.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
#     --method sw \
#     --data_type text \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --subspace_dim 64 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sw_new"

# python ./src/steering/calculate_steering_vectors_ot1.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
#     --method sinkhorn \
#     --use_partial_ot \
#     --data_type text \
#     --n_samples 20 \
#     --threshold 0.85 \
#     --concept_percentile 0.7 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_new_partial_ot"



# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#     --method sinkhorn \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_sanity_check_3"


# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vector_ot_2.py \
#     --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#     --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#     --method sinkhorn \
#     --data_type "activation" \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --max_tokens_sinkhorn 1024 \
#     --n_projections 1000 \
#     --sinkhorn_iter 100 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_sanity_check_2"


# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot.py \
#     --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_sanity_check/glasses_sanity__gs_0.0_prompts_20_neg_attn.pt" \
#     --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_sanity_check/glasses_sanity__gs_0.0_prompts_20_pos_attn.pt" \
#     --method sinkhorn \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_sanity_check_5"

# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot.py \
#     --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_sanity_check_dev/glasses_sanity__gs_4.5_prompts_20_neg_attn.pt" \
#     --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_sanity_check_dev/glasses_sanity__gs_4.5_prompts_20_pos_attn.pt" \
#     --method sinkhorn \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_sanity_check_4"


# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vectors_ot.py \
#     --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_sanity_check/glasses_sanity__gs_0.0_prompts_20_neg_attn.pt" \
#     --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_sanity_check/glasses_sanity__gs_0.0_prompts_20_pos_attn.pt" \
#     --method sinkhorn \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_sanity_check_7"


# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vector_ot_2.py \
#     --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_sanity_check/glasses_sanity__gs_0.0_prompts_20_neg_attn.pt" \
#     --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_sanity_check/glasses_sanity__gs_0.0_prompts_20_pos_attn.pt" \
#     --method sinkhorn \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --data_type "activation" \
#     --save_dir "experiments/flux_schnell/add/final_steering/glasses_sin_sanity_check_8"

# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vector_ot_2.py \
#     --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_dual/glasses__gs_0.0_prompts_20_neg_attn.pt" \
#     --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_dual/glasses__gs_0.0_prompts_20_pos_attn.pt" \
#     --method sinkhorn \
#     --img_mode "ot" \
#     --txt_mode "ot" \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --data_type "dual" \
#     --save_dir "experiments/flux_schnell/add/final_steering/sinhorn_dual"


# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vector_ot_2.py \
#     --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_dual/glasses__gs_0.0_prompts_20_neg_attn.pt" \
#     --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/add/data_vector_dual/glasses__gs_0.0_prompts_20_pos_attn.pt" \
#     --method sinkhorn \
#     --img_mode "meandiff" \
#     --txt_mode "meandiff" \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --data_type "dual" \
#     --save_dir "experiments/flux_schnell/add/final_steering/md_dual"

# CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vector_ot_2.py \
#     --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_add/data_vector/data_vector_attn/glasses__gs_0.0_prompts_20_neg_attn.pt" \
#     --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_add/data_vector/data_vector_attn/glasses__gs_0.0_prompts_20_pos_attn.pt" \
#     --method sw \
#     --txt_mode "meandiff" \
#     --img_mode "meandiff" \
#     --timesteps 4 \
#     --blocks 19 \
#     --n_samples 20 \
#     --epsilon 0.05 \
#     --threshold 0.85 \
#     --data_type "activation" \
#     --save_dir "experiments/flux_schnell/test_compare_add/final_steering/act_txt_meandiff_tokenwise"


CUDA_VISIBLE_DEVICES=0 python ./src/steering/calculate_steering_vector_ot_2.py \
    --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_add/data_vector/data_vector_block/glasses__gs_0.0_prompts_20_neg_block.pt" \
    --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_add/data_vector/data_vector_block/glasses__gs_0.0_prompts_20_pos_block.pt" \
    --method sw \
    --txt_mode "meandiff" \
    --img_mode "meandiff" \
    --timesteps 4 \
    --blocks 19 \
    --n_samples 20 \
    --epsilon 0.05 \
    --threshold 0.85 \
    --data_type "activation" \
    --save_dir "experiments/flux_schnell/test_compare_add/final_steering/act_txt_meandiff_tokenwise_block"
