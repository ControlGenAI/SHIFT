# # 1
# python calculate_steering.py \
#        --pos_path "style_experiments_flux_dev_test/data_vectors/anime/anime__gs_3.5_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_dev_test/data_vectors/anime/anime__gs_3.5_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_dev_test/final_steering/anime/base \
#        --save_svm \
#        --timesteps 20 \
#        --blocks 57 \

# 1
# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell_test/data_vectors/anime/anime__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell_test/data_vectors/anime/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell_test/final_steering/anime/base \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57 \

# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell_test/data_vectors/watercolor/watercolor__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell_test/data_vectors/watercolor/watercolor__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell_test/final_steering/watercolor/base \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57 \


# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell_test/data_vectors/sketch/sketch__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell_test/data_vectors/sketch/sketch__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell_test/final_steering/sketch/base \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57 \


# python calculate_steering.py \
#        --pos_path "style_experiments_flux_dev_test/data_vectors/sketch/sketch__gs_3.5_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_dev_test/data_vectors/sketch/sketch__gs_3.5_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_dev_test/final_steering/sketch/base \
#        --save_svm \
#        --timesteps 20 \
#        --blocks 57 \

# python calculate_steering.py \
#        --pos_path "style_experiments_flux_dev_test/data_vectors/watercolor/watercolor__gs_3.5_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_dev_test/data_vectors/watercolor/watercolor__gs_3.5_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_dev_test/final_steering/watercolor/base \
#        --save_svm \
#        --timesteps 20 \
#        --blocks 57 \




python calculate_steering.py \
       --pos_path "style_experiments_flux_schnell_test/data_vectors/sketch_txt__gs_0.0_prompts_20_pos_attn_enc.pt" \
       --neg_path "style_experiments_flux_schnell_test/data_vectors/sketch_txt__gs_0.0_prompts_20_neg_attn_enc.pt" \
       --n_samples 20 \
       --save_dir style_experiments_flux_schnell_test/final_steering/sketch_bt/base \
       --save_svm \
       --timesteps 4 \
       --blocks 57 \
       --best_tokens \

python calculate_steering.py \
       --pos_path "style_experiments_flux_schnell_test/data_vectors/anime_txt__gs_0.0_prompts_20_pos_attn_enc.pt" \
       --neg_path "style_experiments_flux_schnell_test/data_vectors/anime_txt__gs_0.0_prompts_20_neg_attn_enc.pt" \
       --n_samples 20 \
       --save_dir style_experiments_flux_schnell_test/final_steering/anime_bt/base \
       --save_svm \
       --timesteps 4 \
       --blocks 57 \
       --best_tokens \



# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell/data_vectors/3d/3d__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell/data_vectors/anime/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell/final_steering/3d/bt \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57


# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell/data_vectors/cyberpunk/cyberpunk__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell/data_vectors/anime/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell/final_steering/cyberpunk/bt \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57


# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell/data_vectors/impressionism/impressionism__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell/data_vectors/anime/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell/final_steering/impressionism/bt \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57


# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell/data_vectors/photorealism/photorealism__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell/data_vectors/anime/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell/final_steering/photorealism/bt \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57


# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell/data_vectors/sketch/sketch__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell/data_vectors/anime/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell/final_steering/sketch/bt \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57


# python calculate_steering.py \
#        --pos_path "style_experiments_flux_schnell/data_vectors/watercolor/watercolor__gs_0.0_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_schnell/data_vectors/anime/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_schnell/final_steering/watercolor/bt \
#        --save_svm \
#        --timesteps 4 \
#        --blocks 57

