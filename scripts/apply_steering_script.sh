# # python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "add_experiments_flux_schnell/final_steering/fox_wolf/base" \
# #     --vector_type "diff" \
# #     --task "add concept" \
# #     --strength 5000.0 \
# #     --use_ssim_mask \
# #     --top_k_percent 0.01 \
# #     --orthogonal_projection \
# #     --use_cls \
# #     --min_signal_threshold 0.5 \
# #     --cls_min 20.0 \
# #     --results_dir "./results_debug" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'normals' \
# #     --prompts_path "simple_prompts_animal.txt"


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_0" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 0.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_05" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_1" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# # ###############




# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/sketch/0_1" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/sketch/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/sketch/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # #################

# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/anime/0_1" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/anime/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/anime/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # #################

# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/0_1" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# # ########################

# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 100.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/100_1_diff_s" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 300.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/300_1_diff_s" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 100.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/100_1_diff_m" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'mean' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 300.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/300_1_diff_m" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'mean' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 100.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/100_1_normals_a" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'normals' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# # CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
# #     --task "add concept" \
# #     --strength 300.0 \
# #     --strength_txt 1.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/300_1_normals_s" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'normals' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/100_1_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/300_1_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/100_1_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/300_1_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/100_1_normals_a" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/300_1_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "add_experiments_flux_schnell/final_steering/fox_wolf/base" \
#     --vector_type "diff" \
#     --task "add concept" \
#     --strength 5000.0 \
#     --use_ssim_mask \
#     --top_k_percent 0.01 \
#     --orthogonal_projection \
#     --use_cls \
#     --min_signal_threshold 0.5 \
#     --cls_min 20.0 \
#     --results_dir "./results_debug" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --prompts_path "simple_prompts_animal.txt"


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_1" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_15" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/0_2" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# ###############




# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/0_1" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/0_15" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/0_2" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# #################

# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/0_1" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/0_15" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/0_2" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# #################

# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/0_1" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/0_15" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/0_2" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# ########################

# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/100_1_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/300_1_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/100_1_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/300_1_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/100_1_normals_a" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberkpunk/300_1_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/1000_15_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/300_15_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/1000_15_diff_m_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/300_15_diff_m_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/1000_15_normals_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/300_15_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

### anime


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/1000_15_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/300_15_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/1000_15_diff_m_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/300_15_diff_m_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/1000_15_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/300_15_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

### cyberpunk


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberpunk/1000_15_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberpunk/300_15_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberpunk/1000_15_diff_m_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberpunk/300_15_diff_m_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberpunk/1000_15_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberpunk/300_15_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

### sketch


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/1000_15_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/300_15_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/1000_15_diff_m_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/300_15_diff_m_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/1000_15_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/300_15_normals_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 




CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_9" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 




CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_12" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_7" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


#################################



CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/anime" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/anime" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_9" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 




CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/anime" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_12" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/anime" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_7" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

###########


CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/anime" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/anime/300_15_mean_diff_s_only_pooled" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 




CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/style/generated_images/sketch/300_15_mean_diff_s_only_pooled" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 
