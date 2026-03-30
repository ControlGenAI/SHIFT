

CUDA_VISIBLE_DEVICES=1, python ./src/steering/apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/final_steering/nudity_135" \
    --task "nudity" \
    --strength 250.0 \
    --strength_txt 6.0 \
    --top_k_percent 0.95 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "experiments/flux_schnell/remove/generated_images/coco_512/test10" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --steer_txt \
    --use_cls \
    --remove_prompt " nudity" \
    --prompts_path "/home/jovyan/konovalova/clean_code/steering/all_coco_prompts.txt"

