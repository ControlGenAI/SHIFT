#!/bin/bash
# Calculate steering vectors from extracted activations and text embeddings.

# --- Example: activation-based steering vector (SVM / diff) ---
python ./src/steering/calculate_steering_vectors.py \
    --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/data_vector/data_vector_attn_100/nudity__gs_0.0_prompts_150_neg_block.pt" \
    --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/data_vector/data_vector_attn_100/nudity__gs_0.0_prompts_150_pos_block.pt" \
    --save_svm \
    --timesteps 4 \
    --blocks 19 \
    --n_samples 135 \
    --classifier 'none' \
    --save_dir "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/final_steering/block_steering"

python ./src/steering/calculate_steering_vectors.py \
    --neg_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/data_vector/data_vector_attn_100/nudity__gs_0.0_prompts_150_neg_block.pt" \
    --pos_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/data_vector/data_vector_attn_100/nudity__gs_0.0_prompts_150_pos_block.pt" \
    --method 'diff' \
    --n_samples 135 \
    --classifier 'none' \
    --save_dir "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/final_steering/block_steering"



# python ./src/steering/calculate_steering_vectors.py \
#     --neg_path "experiments/flux_schnell/remove/data_vectors/nudity__gs_0.0_prompts_135_neg_attn_enc.pt" \
#     --pos_path "experiments/flux_schnell/remove/data_vectors/nudity__gs_0.0_prompts_135_pos_attn_enc.pt" \
#     --method 'diff' \
#     --n_namples 135 \
#     --save_dir "experiments/flux_schnell/remove/final_steering/nudity_135/"

# # --- Example: text-embedding-based steering vector ---
# python ./src/steering/calculate_steering_vectors.py \
#     --neg_path "experiments/flux_schnell/remove/data_vectors_txt/people__Nudity_prompts_135_neg_embeddings.pt" \
#     --pos_path "experiments/flux_schnell/remove/data_vectors_txt/people_Nudity_Nudity_prompts_135_pos_embeddings.pt" \
#     --method text \
#     --n_samples 135 \
#     --save_dir "experiments/flux_schnell/remove/final_steering/nudity_135"
