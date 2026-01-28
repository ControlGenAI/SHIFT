# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 25 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob_back/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_back/base_remove_25_no_svm'


# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.5 \
#        --strength 100 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob/base \
#        --t_structure 11110 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_difference/base'

# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 400 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/snoopy_noisy_2/base \
#        --t_structure 11110 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --cls_min 50 \
#        --results_dir 'remove_experiments/generated_images/snoopy_difference/base'


# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.5 \
#        --strength 200 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob/base \
#        --t_structure 11110 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_difference/base'

# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.5 \
#        --strength 100 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob/base \
#        --t_structure 11110 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_difference/base'






# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.2 \
#        --strength 25 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/sketch/base \
#        --t_structure 0 \
#        --block_structure 15 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/sketch/base_remove_25_1_not_cliped_structure_05_15'

# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.8 \
#        --strength 25 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/sketch/base \
#        --t_structure 0 \
#        --block_structure 15 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/sketch/base_remove_25_1_not_cliped_structure_08_15'


# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 15 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/sketch/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/sketch/base_remove_15_1_not_cliped'



# CUDA_VISIBLE_DIVECES=3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob_back/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_back_mick/base_remove_20'


# CUDA_VISIBLE_DIVECES=3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 45 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob_back/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_back_mick/base_remove_45'


# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 10 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/snoopy/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/snoopy/base_remove_10'



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/hat" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/hat/1000_15_mean_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/glasses/1000_15_mean_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/lipstick/1000_15_mean_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/smile" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/smile/1000_15_mean_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/apple" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/apple/1000_15_mean_diff_m" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"





# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/hat" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/hat/1000_15_mean_diff_s_ssim_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/glasses/1000_15_mean_diff_s_ssim_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/lipstick/1000_15_mean_diff_s_ssim_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/smile" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/smile/1000_15_mean_diff_s_ssim_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/apple" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/apple/1000_15_mean_diff_s_ssim_05" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"




CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/apple" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/apple/1000_15_mean_diff_s_only_pooled_0_9_2_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --t_steering "2,3" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/hat" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/hat/1000_15_mean_diff_s_only_pooled_0_9_2_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --t_steering "2,3" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/glasses/1000_15_mean_diff_s_only_pooled_0_9_2_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --t_steering "2,3" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/lipstick/1000_15_mean_diff_s_only_pooled_0_9_2_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --t_steering "2,3" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/smile" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/smile/1000_15_mean_diff_s_only_pooled_0_9_2_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --t_steering "2,3" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/apple" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/apple/1000_15_mean_diff_s_only_pooled_0_12" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/hat" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/hat/1000_15_mean_diff_s_only_pooled_0_12" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/glasses/1000_15_mean_diff_s_only_pooled_0_12" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/lipstick/1000_15_mean_diff_s_only_pooled_0_12" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/smile" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/smile/1000_15_mean_diff_s_only_pooled_0_12" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"

