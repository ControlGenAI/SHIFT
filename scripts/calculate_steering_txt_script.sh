# python calculate_steering_vectors.py \
#    --pos_path "test/data_vectors_txt_2/sketch_txt__prompts_20_pos_embeddings.pt" \
#    --neg_path "test/data_vectors_txt_2/sketch_txt__prompts_20_neg_embeddings.pt" \
#    --method text \
#    --save_dir "test/final_steering/sketch_1"

# python calculate_steering_vectors.py \
#    --pos_path "test/data_vectors_txt_2/style_sketches, pencil drawing_sketches, pencil drawing_prompts_20_pos_embeddings.pt" \
#    --neg_path "test/data_vectors_txt_2/style__sketches, pencil drawing_prompts_20_neg_embeddings.pt" \
#    --method text \
#    --save_dir "test/final_steering/sketch_1"

# python calculate_steering_vectors.py \
#    --neg_path "test/data_vectors_txt_2/style__sketches, pencil_drawing_prompts_20_neg_embeddings.pt" \
#    --pos_path "test/data_vectors_txt_2/style_sketches, pencil_drawing_sketches, pencil_drawing_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "test/final_steering/sketch_1"

# =============== flux-schnell style ====

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/style/data_vectors_txt/style__sketches, pencil drawing_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/style/data_vectors_txt/style_sketches, pencil drawing_sketches, pencil drawing_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/style/final_steering/sketch"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/style/data_vectors_txt/style__anime style, large expressive eyes_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/style/data_vectors_txt/style_anime style, large expressive eyes_anime style, large expressive eyes_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/style/final_steering/anime"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/style/data_vectors_txt/style__cyberpunk, neon lights_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/style/data_vectors_txt/style_cyberpunk, neon lights_cyberpunk, neon lights_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/style/final_steering/cyberpunk"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/style/data_vectors_txt/style__impressionism, Claude Monet_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/style/data_vectors_txt/style_impressionism, Claude Monet_impressionism, Claude Monet_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/style/final_steering/impressionism"


# # ====================== flux schnell add ===========

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors_txt/concrete__big apple_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors_txt/concrete_big apple_big apple_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/add/final_steering/apple"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors_txt/concrete__big hat_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors_txt/concrete_big hat_big hat_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/add/final_steering/hat"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors_txt/concrete__big smile_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors_txt/concrete_big smile_big smile_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/add/final_steering/smile"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors_txt/concrete__long beard_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors_txt/concrete_long beard_long beard_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/add/final_steering/beard"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors_txt/concrete__red lipstick_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors_txt/concrete_red lipstick_red lipstick_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/add/final_steering/lipstick"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/add/final_steering/glasses"


# # ============= flux chnell switch =========

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors_txt/switch_vase_kettle_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors_txt/switch_kettle_kettle_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/switch/final_steering/kettle_vase"


# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors_txt/switch_banana_apple_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors_txt/switch_apple_apple_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/switch/final_steering/apple_banana"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors_txt/switch_bicycle_car_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors_txt/switch_car_car_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/switch/final_steering/car_bicycle"


# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors_txt/switch_dog_cat_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors_txt/switch_cat_cat_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/switch/final_steering/cat_dog"

# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors_txt/switch_man_woman_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors_txt/switch_woman_woman_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/switch/final_steering/gender"


# python calculate_steering_vectors.py \
#    --neg_path "experiments/flux_schnell/switch/data_vectors_txt/switch_young_old_prompts_20_neg_embeddings.pt" \
#    --pos_path "experiments/flux_schnell/switch/data_vectors_txt/switch_old_old_prompts_20_pos_embeddings.pt" \
#    --method text \
#    --save_dir "experiments/flux_schnell/switch/final_steering/age"

python calculate_steering_vectors.py \
   --neg_path "experiments/flux_schnell/remove/data_vectors_txt/concrete__Snoopy_prompts_20_neg_embeddings.pt" \
   --pos_path "experiments/flux_schnell/remove/data_vectors_txt/concrete_Snoopy_Snoopy_prompts_20_pos_embeddings.pt" \
   --method text \
   --save_dir "experiments/flux_schnell/remove/final_steering/snoopy"
