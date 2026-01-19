

CUDA_VISIBLE_DIVECES=0, python apply_steering_with_injection_flux.py \
    --model_name "stabilityai/stable-diffusion-3.5-medium" \
    --data_dir style_experiments/final_steering/cyberpunk/base \
    --task "add concept" \
    --strength 0.0 \
    --strength_txt 0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "results_debug/sd3_test_1" \
    --inference_steps 40 \
    --seed 42 \
    --vector_type 'normals' \
    --steering_type 'separate' \
    --guidance_scale 4.5 \
    --use_cls \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 
