# # ============== diff style ===============
# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/style/data_vectors/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/style/data_vectors/anime__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/style/final_steering/anime/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/style/data_vectors/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/style/data_vectors/cyberpunk__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/style/final_steering/cyberpunk/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/style/data_vectors/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/style/data_vectors/impressionism__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/style/final_steering/impressionism/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/style/data_vectors/anime__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/style/data_vectors/sketch__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/style/final_steering/sketch/"

# # ============== diff add ===============
# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors/apple__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors/apple__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/add/final_steering/apple/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors/beard__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors/beard__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/add/final_steering/beard/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors/glasses__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/add/final_steering/glasses/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors/hat__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors/hat__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/add/final_steering/hat/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors/lipstick__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors/lipstick__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/add/final_steering/lipstick/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors/smile__gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors/smile__gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/add/final_steering/smile/"


# # ============== diff switch ===============

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors/age_young_gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors/age_young_gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/switch/final_steering/age/"


# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors/apple_banana_banana_gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors/apple_banana_banana_gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'rbf' \
#    --save_dir "experiments/flux_schnell/switch/final_steering/apple_banana_logreg/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors/car_bicycle_bicycle_gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors/car_bicycle_bicycle_gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/switch/final_steering/car_bicycle/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors/cat_dog_dog_gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors/cat_dog_dog_gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/switch/final_steering/cat_dog/"


# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors/gender_man_gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors/gender_man_gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/switch/final_steering/gender/"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors/kettle_vase_vase_gs_0.0_prompts_20_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors/kettle_vase_vase_gs_0.0_prompts_20_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'rbf' \
#    --save_dir "experiments/flux_schnell/switch/final_steering/kettle_vase_rbf_04/"

# # python calculate_steering_vectors.py \
# #    --pos_path "test/data_vectors_txt/pos_text.pt" \
# #    --neg_path "test/data_vectors_txt/neg_text.pt" \
# #    --method text \
# #    --save_dir "test/final_steering/sketch"

# # python calculate_steering_vectors.py \
# #    --pos_path "style_experiments_flux_schnell_test/data_vectors/sketch_txt__gs_0.0_prompts_20_pos_attn_enc.pt" \
# #    --neg_path "style_experiments_flux_schnell_test/data_vectors/sketch_txt__gs_0.0_prompts_20_neg_attn_enc.pt" \
# #    --save_dir "test/final_steering/sketch/base" \
# #    --save_svm \
# #    --timesteps 4 \
# #    --blocks 19 \
# #    --classifier 'none' \


# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/remove/data_vectors/nudity__gs_0.0_prompts_135_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/remove/data_vectors/nudity__gs_0.0_prompts_135_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --n_samples 135 \
#    --classifier 'none' \
#    --save_dir "experiments/flux_schnell/remove/final_steering/nudity_135/"


# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors_image/apple_banana_banana_gs_0.0_prompts_63_neg_attn_enc.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors_image/apple_banana_banana_gs_0.0_prompts_63_pos_attn_enc.pt" \
#    --save_svm \
#    --timesteps 4 \
#    --blocks 19 \
#    --classifier 'rbf' \
#    --save_dir "experiments/flux_schnell/switch/final_steerin_image/apple_banana_rbf_63_PCA_20/"


python calculate_steering_vectors.py \
   --neg_path "experiments/flux_schnell/switch/data_vectors_image/apple_banana_banana_gs_0.0_prompts_63_neg_attn_enc.pt" \
   --pos_path "experiments/flux_schnell/switch/data_vectors_image/apple_banana_banana_gs_0.0_prompts_63_pos_attn_enc.pt" \
   --save_svm \
   --timesteps 4 \
   --blocks 19 \
   --classifier 'none' \
   --save_dir "experiments/flux_schnell/switch/final_steerin_image/apple_banana_63_PCA_20/"


python calculate_steering_vectors.py \
   --neg_path "experiments/flux_schnell/switch/data_vectors_image/apple_banana_banana_gs_0.0_prompts_63_neg_attn_enc.pt" \
   --pos_path "experiments/flux_schnell/switch/data_vectors_image/apple_banana_banana_gs_0.0_prompts_63_pos_attn_enc.pt" \
   --save_svm \
   --timesteps 4 \
   --blocks 19 \
   --classifier 'logistic' \
   --save_dir "experiments/flux_schnell/switch/final_steerin_image/apple_banana_logreg_63_PCA_20/"
