# # CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
# #        --structure 1.0 \
# #        --strength 0.0 \
# #        --n_samples 20 \
# #        --data_dir remove_experiments_flux_schnell/final_steering/mickey/base \
# #        --t_structure 1500 \
# #        --block_structure 0 \
# #        --block_steering 'all' \
# #        --model_name black-forest-labs/FLUX.1-schnell \
# #        --inference_steps 4 \
# #        --save_svm \
# #        --guidance_scale 0.0 \
# #        --cls_min 21 \
# #        --block_steering 'all' \
# #        --task 'remove' \
# #        --results_dir 'remove_experiments_flux_schnell/generated_images/mickey_txt/test'

# CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
#        --structure 0.5 \
#        --strength 0.0 \
#        --n_samples 20 \
#        --data_dir remove_experiments_flux_schnell/final_steering/spongebob/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-schnell \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 0.0 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --task 'remove' \
#        --results_dir 'remove_experiments_flux_schnell/generated_images/test/test'

# # CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
# #        --structure 0.5 \
# #        --strength 0.0 \
# #        --n_samples 20 \
# #        --data_dir remove_experiments_flux_schnell/final_steering/spongebob/base \
# #        --t_structure 1500 \
# #        --block_structure 0 \
# #        --block_steering 'all' \
# #        --model_name black-forest-labs/FLUX.1-schnell \
# #        --inference_steps 4 \
# #        --save_svm \
# #        --guidance_scale 0.0 \
# #        --cls_min 21 \
# #        --block_steering 'all' \
# #        --task 'remove' \
# #        --results_dir 'remove_experiments_flux_schnell/generated_images/Spongebob_txt/test'





# # CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
# #        --structure 0.5 \
# #        --strength 100.0 \
# #        --n_samples 20 \
# #        --data_dir remove_experiments_flux_schnell/final_steering/spongebob/base \
# #        --t_structure 1500 \
# #        --block_structure 0 \
# #        --block_steering 'all' \
# #        --model_name black-forest-labs/FLUX.1-schnell \
# #        --inference_steps 4 \
# #        --save_svm \
# #        --guidance_scale 0.0 \
# #        --cls_min 21 \
# #        --block_steering 'all' \
# #        --task 'remove' \
# #        --results_dir 'remove_experiments_flux_schnell/generated_images/spongebob_txt/test'

CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/remove/final_steering/snoopy" \
    --task "remove" \
    --strength 100.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/remove/generated_images/snoopy/100_20_mean_diff_s_only_pooled_1_9" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --steer_txt \
    --remove_prompt " Mickey" \
    --block_steering "0,1,2,3,4,5,6,7,8,9" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_remove.txt"

CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/remove/final_steering/snoopy" \
    --task "remove" \
    --strength 500.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/remove/generated_images/snoopy/500_20_mean_diff_s_not_pooled" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --steer_txt \
    --remove_prompt " SpongeBob" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_remove.txt"



CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/remove/final_steering/snoopy" \
    --task "remove" \
    --strength 500.0 \
    --strength_txt 2.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/remove/generated_images/snoopy/500_20_mean_diff_s_not_pooled" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --steer_txt \
    --remove_prompt " legislator" \
    --prompts_path "prompts_collection/ablation/ablation_prompts_remove.txt"

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/remove/final_steering/snoopy" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/remove/generated_images/snoopy/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --steer_txt \
#     --remove_prompt " Mickey" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_remove.txt"


# CUDA_VISIBLE_DEVICES=3,1, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/remove/final_steering/snoopy" \
#     --task "remove" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.5 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/remove/generated_images/snoopy/0_0" \
#     --inference_steps 4 \
#     --seed 42 \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --steer_txt \
#     --remove_prompt " legislator" \
#     --prompts_path "prompts_collection/ablation/ablation_prompts_remove.txt"
