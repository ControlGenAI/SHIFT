# python calculate_steering.py \
#        --pos_path "remove_experiments_flux_schnell/data_vectors/Snoopy_realistic_gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "remove_experiments_flux_schnell/data_vectors/Snoopy_realistic_gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir remove_experiments_flux_schnell/final_steering/snoopy/base \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 19 \

python calculate_steering.py \
       --pos_path "remove_experiments_flux_schnell/data_vectors/Spongebob_realistic_gs_0.0_prompts_20_pos_attn_enc.pt" \
       --neg_path "remove_experiments_flux_schnell/data_vectors/Spongebob_realistic_gs_0.0_prompts_20_neg_attn_enc.pt" \
       --n_samples 20 \
       --save_dir remove_experiments_flux_schnell/final_steering/spongebob/base \
       --save_svm \
       --timesteps 4 \
       --blocks 19 \

# python calculate_steering.py \
#        --pos_path "remove_experiments_flux_schnell/data_vectors/Mickey_realistic_gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "remove_experiments_flux_schnell/data_vectors/Mickey_realistic_gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir remove_experiments_flux_schnell/final_steering/mickey/base \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 19 \