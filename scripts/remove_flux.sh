# # CUDA_VISIBLE_DEVICES=0, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-dev" \
# #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# #     --task "remove" \
# #     --strength 500.0 \
# #     --strength_txt 6.0 \
# #     --top_k_percent 0.95 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_dev/remove/generated_images/nudity_512/500_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512_use_cls_correct" \
# #     --inference_steps 28 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 3.5 \
# #     --steer_txt \
# #     --use_cls \
# #     --remove_prompt " nudity" \
# #     --prompts_path "prompts_collection/nudity_flux.txt"





# # CUDA_VISIBLE_DEVICES=0, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-dev" \
# #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# #     --task "remove" \
# #     --strength 100.0 \
# #     --strength_txt 8.0 \
# #     --top_k_percent 0.95 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_dev/remove/generated_images/nudity_512/100_8_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512" \
# #     --inference_steps 28 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 3.5 \
# #     --steer_txt \
# #     --remove_prompt " nudity" \
# #     --use_cls \
# #     --prompts_path "prompts_collection/nudity_flux.txt"


# # CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-dev" \
# #     --data_dir "experiments/flux_dev/remove/final_steering/nudity_135" \
# #     --task "remove" \
# #     --strength 250.0 \
# #     --strength_txt 6.0 \
# #     --top_k_percent 0.95 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_dev/remove/generated_images/nudity_512_dev/250_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512" \
# #     --inference_steps 28 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 3.5 \
# #     --steer_txt \
# #     --remove_prompt " nudity" \
# #     --prompts_path "prompts_collection/nudity_flux.txt"


# # CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-dev" \
# #     --data_dir "experiments/flux_dev/remove/final_steering/nudity_135" \
# #     --task "remove" \
# #     --strength 100.0 \
# #     --strength_txt 6.0 \
# #     --top_k_percent 0.95 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_dev/remove/generated_images/nudity_512_dev/100_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512" \
# #     --inference_steps 28 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 3.5 \
# #     --steer_txt \
# #     --remove_prompt " nudity" \
# #     --prompts_path "prompts_collection/nudity_flux.txt"


# # #################



# # # CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-dev" \
# # #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# # #     --task "remove" \
# # #     --strength 500.0 \
# # #     --strength_txt 6.0 \
# # #     --top_k_percent 0.95 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_dev/remove/generated_images/nudity_512/500_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512" \
# # #     --inference_steps 28 \
# # #     --seed 42 \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 3.5 \
# # #     --steer_txt \
# # #     --remove_prompt " nudity" \
# # #     --prompts_path "prompts_collection/nudity_flux.txt"


# # # CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-dev" \
# # #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# # #     --task "remove" \
# # #     --strength 250.0 \
# # #     --strength_txt 6.0 \
# # #     --top_k_percent 0.95 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_dev/remove/generated_images/nudity_512/250_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512" \
# # #     --inference_steps 28 \
# # #     --seed 42 \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 3.5 \
# # #     --steer_txt \
# # #     --remove_prompt " nudity" \
# # #     --prompts_path "prompts_collection/nudity_flux.txt"


# # # CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-dev" \
# # #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# # #     --task "remove" \
# # #     --strength 100.0 \
# # #     --strength_txt 6.0 \
# # #     --top_k_percent 0.95 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_dev/remove/generated_images/nudity_512/100_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512" \
# # #     --inference_steps 28 \
# # #     --seed 42 \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 3.5 \
# # #     --steer_txt \
# # #     --remove_prompt " nudity" \
# # #     --prompts_path "prompts_collection/nudity_flux.txt"


# # CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-dev" \
# #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# #     --task "remove" \
# #     --strength 500.0 \
# #     --strength_txt 6.0 \
# #     --top_k_percent 0.95 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_dev/remove/generated_images/coco_512/500_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512_cls" \
# #     --inference_steps 28 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 3.5 \
# #     --steer_txt \
# #     --remove_prompt " nudity" \
# #     --use_cls \
# #     --prompts_path "/home/jovyan/konovalova/steering/all_coco_prompts.txt"



# # CUDA_VISIBLE_DEVICES=0, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-dev" \
# #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# #     --task "remove" \
# #     --strength 250.0 \
# #     --strength_txt 6.0 \
# #     --top_k_percent 0.95 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_dev/remove/generated_images/coco_512/250_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512_cls" \
# #     --inference_steps 28 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 3.5 \
# #     --steer_txt \
# #     --remove_prompt " nudity" \
# #     --use_cls \
# #     --prompts_path "/home/jovyan/konovalova/steering/all_coco_prompts.txt"




# CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_dev/remove/final_steering/nudity_135" \
#     --task "remove" \
#     --strength 250.0 \
#     --strength_txt 6.0 \
#     --top_k_percent 0.95 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/remove/generated_images/coco_512_dev/250_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512_cls" \
#     --inference_steps 28 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --steer_txt \
#     --remove_prompt " nudity" \
#     --use_cls \
#     --prompts_path "/home/jovyan/konovalova/steering/all_coco_prompts.txt"


# CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_dev/remove/final_steering/kelly_prompts_20" \
#     --task "remove" \
#     --strength 250.0 \
#     --strength_txt 5 \
#     --top_k_percent 0.95 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/remove/generated_images/kelly_prompts_20/250_5_cls_correct_3" \
#     --inference_steps 28 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --steer_txt \
#     --use_cls \
#     --remove_prompt " nudity" \
#     --prompts_path "niche_art_prompts.txt"


# CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-dev" \
#     --data_dir "experiments/flux_schnell/remove/final_steering/kelly_prompts_20" \
#     --task "remove" \
#     --strength 250.0 \
#     --strength_txt 5 \
#     --top_k_percent 0.95 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_dev/remove/generated_images/kelly_prompts_20_dev/250_5_cls_correct_3" \
#     --inference_steps 28 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 3.5 \
#     --steer_txt \
#     --use_cls \
#     --remove_prompt " nudity" \
#     --prompts_path "niche_art_prompts.txt"


CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-dev" \
    --data_dir "experiments/flux_dev/remove/final_steering/kelly_prompts_20" \
    --task "remove" \
    --strength 250.0 \
    --strength_txt 5 \
    --top_k_percent 0.95 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_dev/remove/generated_images/kelly_prompts_20/250_5_cls_correct_3" \
    --inference_steps 28 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 3.5 \
    --steer_txt \
    --use_cls \
    --remove_prompt " nudity" \
    --prompts_path "niche_art_prompts.txt"
