python calculate_steering.py \
       --pos_path "remove_experiments/data_vectors/snoopy_remove_realistic_gs_4.5_prompts_20_pos_attn_enc.pt" \
       --neg_path "remove_experiments/data_vectors/snoopy_remove_realistic_gs_4.5_prompts_20_neg_attn_enc.pt" \
       --n_samples 20 \
       --save_dir remove_experiments/final_steering/snoopy_noisy_2/base \
       --save_svm \

# python calculate_steering.py \
#        --pos_path "remove_experiments/data_vectors/sketch_realistic_gs_4.5_prompts_20_pos_attn_enc.pt" \
#        --neg_path "remove_experiments/data_vectors/sketch_realistic_gs_4.5_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir remove_experiments/final_steering/sketch/base \
#        --save_svm \




# python calculate_steering.py \
#        --pos_path "remove_experiments/data_vectors/snoopy back_realistic_gs_4.5_prompts_20_pos_attn_enc.pt" \
#        --neg_path "remove_experiments/data_vectors/snoopy back_realistic_gs_4.5_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir remove_experiments/final_steering/snoopy_back/base \
#        --save_svm \