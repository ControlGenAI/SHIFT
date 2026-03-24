# # CUDA_VISIBLE_DEVICES=0, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# #     --task "remove" \
# #     --strength 250.0 \
# #     --strength_txt 6.0 \
# #     --top_k_percent 0.95 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/remove/generated_images/nudity_512/250_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512_use_cls_correct" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --steer_txt \
# #     --use_cls \
# #     --remove_prompt " nudity" \
# #     --prompts_path "prompts_collection/nudity_flux.txt"


# # CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
# #     --model_name "black-forest-labs/FLUX.1-schnell" \
# #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# #     --task "remove" \
# #     --strength 250.0 \
# #     --strength_txt 6.0 \
# #     --top_k_percent 0.95 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/remove/generated_images/coco_512/250_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --steer_txt \
# #     --remove_prompt " nudity" \
# #     --prompts_path "/home/jovyan/konovalova/steering/all_coco_prompts.txt"



# # # CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
# # #     --model_name "black-forest-labs/FLUX.1-schnell" \
# # #     --data_dir "experiments/flux_schnell/remove/final_steering/nudity_135" \
# # #     --task "remove" \
# # #     --strength 100.0 \
# # #     --strength_txt 6.0 \
# # #     --top_k_percent 0.95 \
# # #     --min_signal_threshold 0.05 \
# # #     --cls_min 20.0 \
# # #     --results_dir "experiments/flux_schnell/remove/generated_images/nudity_512/250_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2_512" \
# # #     --inference_steps 4 \
# # #     --seed 42 \
# # #     --vector_type 'diff' \
# # #     --steering_type 'separate' \
# # #     --guidance_scale 0.0 \
# # #     --steer_txt \
# # #     --remove_prompt " nudity" \
# # #     --prompts_path "prompts_collection/nudity_flux.txt"



# CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/remove/final_steering/snoopy_2" \
#     --task "remove" \
#     --strength 500.0 \
#     --strength_txt 6.0 \
#     --top_k_percent 0.1 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_normals_s_one_vec_new_only_pooled_2_scaled_clip" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'normals' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --steer_txt \
#     --remove_prompt " legislator" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_remove.txt"



# CUDA_VISIBLE_DEVICES=1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/remove/final_steering/snoopy_2" \
#     --task "remove" \
#     --strength 500.0 \
#     --strength_txt 6.0 \
#     --top_k_percent 0.1 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_m_one_vec_new_only_pooled_2_scaled_clip" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'mean' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --steer_txt \
#     --remove_prompt " legislator" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_remove.txt"


# # CUDA_VISIBLE_DEVICES=0, python apply_steering_with_injection_flux.py \ 500_6_mean_normals_s_one_vec_new_only_pooled_2_scaled_clip
# #     --model_name "black-forest-labs/FLUX.1-dev" \
# #     --data_dir "experiments/flux_schnell/remove/final_steering/snoopy_2" \
# #     --task "remove" \
# #     --strength 500.0 \
# #     --strength_txt 6.0 \
# #     --top_k_percent 0.1 \
# #     --min_signal_threshold 0.05 \
# #     --cls_min 20.0 \
# #     --results_dir "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6" \
# #     --inference_steps 4 \
# #     --seed 42 \
# #     --vector_type 'diff' \
# #     --steering_type 'separate' \
# #     --guidance_scale 0.0 \
# #     --use_cls \
# #     --steer_txt \
# #     --remove_prompt " legislator" \
# #     --prompts_path "prompts_collection/ablation/ablation_prompts_remove.txt"


CUDA_VISIBLE_DEVICES=0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/remove/final_steering/gogh_prompts_3" \
    --task "remove" \
    --strength 0.0 \
    --strength_txt 0.00001 \
    --top_k_percent 0.95 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/remove/generated_images/gogh_prompts_2/0_0" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --steer_txt \
    --use_cls \
    --remove_prompt " nudity" \
    --prompts_path "big_artists_prompts.txt"

