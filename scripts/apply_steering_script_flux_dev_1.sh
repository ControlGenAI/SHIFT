
# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_schnell/add/final_steering/hat" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.25 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/add/generated_images/hat/1000_1_mean_only_pooled_diff_s_0_9" \
#     --inference_steps 50 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.25 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/add/generated_images/glasses/1000_1_mean_only_pooled_diff_s_0_9" \
#     --inference_steps 50 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_schnell/add/final_steering/lipstick" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.25 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/add/generated_images/lipstick/1000_1_mean_only_pooled_diff_s_0_9" \
#     --inference_steps 50 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_schnell/add/final_steering/smile" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.25 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/add/generated_images/smile/1000_1_mean_only_pooled_diff_s_0_9" \
#     --inference_steps 50 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"

# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_schnell/add/final_steering/apple" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.25 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/add/generated_images/apple/1000_1_mean_only_pooled_diff_s_0_9" \
#     --inference_steps 50 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_add.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.25 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/switch/generated_images/age/1000_3_mean_only_pooled_diff_s_0_7" \
#     --inference_steps 50 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-dev" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "add concept" \
    --strength 300.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_dev/style/generated_images/sketch/300_15_mean_diff_s_only_pooled" \
    --inference_steps 50 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 3.5 \
    --use_cls \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 
