CUDA_VISIBLE_DEVICES=1,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/glasses" \
    --task "add concept" \
    --strength 1.0 \
    --strength_txt 0.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/glasses/test_1" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --prompts_path "prompts_collection/dataset_creation/dataset_prompts_add.txt"


CUDA_VISIBLE_DEVICES=2,0, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/add/final_steering/hat" \
    --task "add concept" \
    --strength 0.0 \
    --strength_txt 1.0 \
    --top_k_percent 0.5 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/add/generated_images/hat/check_train_data_not_normed_0_1_not_mean" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --prompts_path "prompts_collection/dataset_creation/dataset_prompts_add.txt"


