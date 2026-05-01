

CUDA_VISIBLE_DEVICES=0, python ./src/steering/apply_steering_with_injection_flux_1.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/final_steering/block_steering_cleaned" \
    --task "remove" \
    --strength 45.0 \
    --strength_txt 0.0 \
    --strength_img 0.0 \
    --top_k_percent 0.95 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --injection_point "block" \
    --results_dir "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_remove/generated_images_big/subspace/block/txt_block_45_txt_pooled_6_cls_coco_cleaned" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --width 512 \
    --height 512 \
    --use_cls \
    --steer_txt \
    --strength_txt 6.0 \
    --prompts_path "/home/jovyan/konovalova/steering/all_coco_prompts.txt" \
