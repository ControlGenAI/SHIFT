# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/glasses/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/add/glasses.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/hat/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/add/hat.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/apple/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/add/apple.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/lipstick/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/add/lipstick.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/smile" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/add/generated_images/smile/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/add/smile.txt"



# #################################   switch ############################

# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/add/final_steering/cat_dog" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/switch/cat.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/apple_banana" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/apple_banana/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/switch/apple.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/kettle_vase" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/kettle_vase/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/switch/kettle.txt"


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/age" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/age/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/switch/age.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/gender" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/gender/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/switch/smile.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/switch/final_steering/car_bicycle" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/switch/generated_images/car_bicycle/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/switch/car.txt"


# ################ style ##################


# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/anime/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/style/anime.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/sketch/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/style/sketch.txt"



# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/cyberpunk" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/style/cyberpunk.txt"




# CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/impressionism" \
#     --task "add concept" \
#     --strength 0.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "experiments/flux_schnell/style/generated_images/impressionism/0_0_target" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --block_steering 9999 \
#     --t_steering "10" \
#     --prompts_path "prompts_collection/ablation/style/impressionism.txt"



CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/switch/final_steering/cat_dog" \
    --task "add concept" \
    --strength 0.0 \
    --strength_txt 0.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/switch/generated_images/cat_dog/0_0_target" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --block_steering 9999 \
    --t_steering "10" \
    --prompts_path "prompts_collection/ablation/switch/cat.txt"

