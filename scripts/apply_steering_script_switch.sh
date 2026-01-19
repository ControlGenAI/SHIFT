# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/0_0" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 0.5 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/0_05" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 1.0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/0_1" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# # ###############

CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 0.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/0_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"

CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_2_diff_s" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"



CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_2_diff_m" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'mean' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_2_diff_s_ssim" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'mean' \
    --guidance_scale 0.0 \
    --use_cls \
    --use_ssim_mask \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"

CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 10000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/10000_2_diff_m" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'mean' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 10000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/10000_2_diff_s_ssim" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'mean' \
    --guidance_scale 0.0 \
    --use_cls \
    --use_ssim_mask \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


###

CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 0.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/0_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"

CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/1000_2_diff_s" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"



CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/1000_2_diff_m" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'mean' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"

CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 10000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/10000_2_diff_m" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'mean' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"


CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/1000_2_diff_s_ssim" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'mean' \
    --guidance_scale 0.0 \
    --use_cls \
    --use_ssim_mask \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"

CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
    --task "add concept" \
    --strength 10000.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/10000_2_diff_s_ssim" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'mean' \
    --guidance_scale 0.0 \
    --use_cls \
    --use_ssim_mask \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 1.0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/0_1" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"



# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 0.5 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/0_05" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


# # # #################

# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 0.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/0_0" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"



# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 1.0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/0_1" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 1.0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/0_1" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 0.5 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/0_05" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_big_objects.txt"

# # #########


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 1.0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/gender/0_1" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 0.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/gender/0_0" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 0.5 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/gender/0_05" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/gender/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/gender/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# # # #################


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 1.0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/age/0_1" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 0.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/age/0_0" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 0.5 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/age/0_05" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 1.5 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/age/0_15" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


# # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 2.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/age/0_2" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# # #########

# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/smile" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 1.0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/smile/0_1" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# # # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # # #     --data_dir "experiments/flux_schnell/switch/final_steering/smile" \
# # # #     --task "add concept" \
# # # #     --strength 0.0 \
# # # #     --strength_txt 0.5 \
# # # #     --top_k_percent 0.01 \
# # # #     --min_signal_threshold 0.05 \
# # # #     --cls_min 20.0 \
# # # #     --results_dir "experiments/flux_schnell/switch/generated_images/smile/0_05" \
# # # #     --inference_steps 4 \
# # # #     --seed 42 \
# # # #     --steer_txt \
# # # #     --vector_type 'diff' \
# # # #     --steering_type 'separate' \
# # # #     --guidance_scale 0.0 \
# # # #     --use_cls \
# # # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/smile" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 1.5 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/smile/0_15" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# # # CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/switch/final_steering/smile" \
# # #     --task "add concept" \
# # #     --strength 0.0 \
# # #     --strength_txt 2.0 \
# # #     --top_k_percent 0.01 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/switch/generated_images/smile/0_2" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --steer_txt \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --use_cls \
# # #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# # # # #################

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/100_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/300_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/100_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/300_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# ###

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/100_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/300_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/100_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/300_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


# ###########################



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/100_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/300_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/1000_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/100_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/300_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/1000_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# ###

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/100_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/300_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/1000_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/100_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/300_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/1000_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# ###########################



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/100_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/300_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/100_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/300_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_2_diff_s" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"

# ###

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/100_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/300_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"



# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 100.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/100_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"

# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 300.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/300_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# CUDA_VISIBLE_DEVICES=0,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 2.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_2_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"
