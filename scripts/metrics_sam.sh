CUDA_VISIBLE_DEVICES=2, python metrics/sam_clip.py \
    --image_dir "experiments/flux_schnell/add/generated_images/glasses/0_1/steered" \
    --save_dir "experiments/flux_schnell/add/generated_images/glasses/0_1/sam" \
    --target_prompt "glasses" \
    --clip_threshold 0.8 \
    --visualize