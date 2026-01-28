

# CUDA_VISIBLE_DIVECES=0, python apply_steering_with_injection_flux.py \
#     --model_name "stabilityai/stable-diffusion-3.5-medium" \
#     --data_dir style_experiments/final_steering/cyberpunk/base \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "results_debug/sd3_test_1" \
#     --inference_steps 40 \
#     --seed 42 \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 4.5 \
#     --use_cls \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 



####################


# CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/0_3_mean" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"



# CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.1 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_ssim_01" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

# CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/0_3_mean" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"



# CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"

# CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.1 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_ssim_01" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


# # CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
# #     --task "add concept" \
# #     --strength 0.0 \
# #     --strength_txt 3.0 \
# #     --top_k_percent 0.01 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/0_3_mean" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --steer_txt \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"



# CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/1000_3_mean_diff_s_ssim" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"


# CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.1 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/1000_3_mean_diff_s_ssim_01" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --use_ssim_mask \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_animals.txt"




# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_only_pooled" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"



# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_only_pooled_0_12" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"




# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_only_pooled_0_9" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"



# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_only_pooled_0_7" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


####



# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_only_pooled" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"



# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_only_pooled_0_12" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"




# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_only_pooled_0_9" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7,8,9" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"



# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 1000.0 \
#     --strength_txt 3.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_only_pooled_0_7" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering "0,1,2,3,4,5,6,7" \
#     --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


###

CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_3_mean_diff_s_only_pooled" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"



CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_3_mean_diff_s_only_pooled_0_12" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9,10,11,12" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"




CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_3_mean_diff_s_only_pooled_0_9" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"



CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_3_mean_diff_s_only_pooled_0_7" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"




CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_3_mean_diff_s_only_pooled_0_7_0" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --t_steering "0" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_3_mean_diff_s_only_pooled_0_7_0_1" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --t_steering "0,1" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/1000_3_mean_diff_s_only_pooled_0_7_2_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --t_steering "2,3" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_small_objects.txt"


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/age" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_only_pooled_0_7_2_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --t_steering "2,3" \
    --block_steering "0,1,2,3,4,5,6,7" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/age" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_only_pooled_0_7_0_1" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --t_steering "0,1" \
    --block_steering "0,1,2,3,4,5,6,7" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/age" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/age/1000_3_mean_diff_s_only_pooled_0_7_0" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --t_steering "0" \
    --block_steering "0,1,2,3,4,5,6,7" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people_age.txt"

CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_only_pooled_0_7_0" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --t_steering "0" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_only_pooled_0_7_0_1" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --t_steering "0,1" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"

CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 3.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/gender/1000_3_mean_diff_s_only_pooled_0_7_2_3" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering "0,1,2,3,4,5,6,7" \
    --t_steering "2,3" \
    --prompts_path "/workspace-SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs_clean/prompts_collection/ablation/ablation_prompts_people.txt"


