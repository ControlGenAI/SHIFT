# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/1000_3_mean_diff_s_only_pooled_0_7_one_vec" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/kettle_vase" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/kettle_vase/1000_3_mean_diff_s_only_pooled_0_7_one_vec" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects_1.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_only_pooled_0_7_one_vec" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_3_mean_diff_s_only_pooled_0_7_one_vec" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_only_pooled_0_7_one_vec" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/1000_15_mean_diff_s_only_pooled" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"



CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/1000_15_mean_diff_s_only_pooled_0_7" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"




CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/1000_15_mean_diff_s_only_pooled_0_9" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"



CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 1.5 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/1000_15_mean_diff_s_only_pooled_0_12" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 1.5 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/1000_15_mean_diff_s_only_pooled_0_7_one_vec" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"

