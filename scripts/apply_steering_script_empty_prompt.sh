CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/apple" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/apple/1000_15_mean_diff_s_0_9_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/apple" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/apple/1000_15_mean_diff_s_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


###

CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/smile" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/smile/1000_15_mean_diff_s_0_9_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/smile" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/smile/1000_15_mean_diff_s_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"

###

CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/lipstick/1000_15_mean_diff_s_0_9_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/lipstick/1000_15_mean_diff_s_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


###

CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/hat" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/hat/1000_15_mean_diff_s_0_9_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/hat" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/hat/1000_15_mean_diff_s_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


###

CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/glasses/1000_15_mean_diff_s_0_9_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/glasses/1000_15_mean_diff_s_empty_prompt" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/apple" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/apple/0_0_empty_prompt" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"
