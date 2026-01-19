# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/0_1" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 0.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/0_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/100_15" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/100_15_d_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type '' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/100_15_n_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 15 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_glasses/1000_5_normed" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with big glasses" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 2.5 \
#     --top_k_percent 0.1 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_glasses/test_01_25_sep_01_005" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with glasses" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 2.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_glasses/test_01_25_separate_05_005" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with glasses" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 2.5 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_glasses/test_005_25_separate_005_005" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with glasses" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 2.5 \
#     --top_k_percent 0.1 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_glasses/test_005_25_separate_01_005" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with glasses" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/hat" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_hat/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with hat" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/hat" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 2.5 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_hat/test_025_25_separate_005_005" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with hat" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/hat" \
    --task "remove" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.1 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/remove_hat/test_025_15_separate_01_005" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --use_ssim_mask \
    --steer_txt \
    --remove_prompt " with hat" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


#####################################


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/apple" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_apple/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with apple" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/apple" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 2.5 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_apple/test_025_25_separate_005_005" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with apple" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/apple" \
    --task "remove" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.1 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/remove_apple/test_015_25_separate_01_005" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --use_ssim_mask \
    --steer_txt \
    --remove_prompt " with apple" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"

#############################



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/smile" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_smile/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with smile" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/smile" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 2.5 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_smile/test_025_25_separate_005_005" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with smile" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/smile" \
    --task "remove" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.1 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/remove_smile/test_025_15_separate_01_005" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --use_ssim_mask \
    --steer_txt \
    --remove_prompt " with smile" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"

#############################



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_lipstick/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with red lipstick" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 2.5 \
#     --top_k_percent 0.05 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_lipstick/test_025_25_separate_005_005" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --steer_txt \
#     --remove_prompt " with red lipstick" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
    --task "remove" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.1 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/remove_lipstick/test_025_15_separate_01_005" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --use_ssim_mask \
    --steer_txt \
    --remove_prompt " with red lipstick" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"

#############################


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 15 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/remove_glasses/100_15_normed" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with big glasses" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 10 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/0_10_normed" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/1000_1" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/100_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/1000_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 0.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/100_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 0.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/1000_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 1000.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/1000_10_no_cls" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/100_10_no_cls" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/remove_sketch/1000_10_no_cls" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 
