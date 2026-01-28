############################ blocks ########################


################################# cyberpunk
python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/cyberpunk/300_1_mean_diff_s_only_pooled/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/cyberpunk/300_1_mean_diff_s_only_pooled/" \
                       --style_prompt "cyberpunk, neon lights" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"


python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/cyberpunk/300_1_mean_diff_s_only_pooled_0_7/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/cyberpunk/300_1_mean_diff_s_only_pooled_0_7/" \
                       --style_prompt "cyberpunk, neon lights" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"


python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/cyberpunk/300_1_mean_diff_s_only_pooled_0_9/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/cyberpunk/300_1_mean_diff_s_only_pooled_0_9/" \
                       --style_prompt "cyberpunk, neon lights" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"

python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/cyberpunk/300_1_mean_diff_s_only_pooled_0_12/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/cyberpunk/300_1_mean_diff_s_only_pooled_0_12/" \
                       --style_prompt "cyberpunk, neon lights" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"

#################################### sketch

python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled/" \
                       --style_prompt "sketches, pencil drawing" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"


python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_7/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_7/" \
                       --style_prompt "sketches, pencil drawing" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"


python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_9/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_9/" \
                       --style_prompt "sketches, pencil drawing" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"

python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_12/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/sketch/300_1_mean_diff_s_only_pooled_0_12/" \
                       --style_prompt "cyberpunk, neon lights" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"

# ##################################### anime

python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled/" \
                       --style_prompt "anime style, large expressive eyes" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"


python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_7/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_7/" \
                       --style_prompt "anime style, large expressive eyes" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"


python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_9/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_9/" \
                       --style_prompt "anime style, large expressive eyes" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"

python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_12/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/anime/300_1_mean_diff_s_only_pooled_0_12/" \
                       --style_prompt "anime style, large expressive eyes" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"

#################################### impressionism

python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/impressionism/300_1_mean_diff_s_only_pooled/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/impressionism/300_1_mean_diff_s_only_pooled/" \
                       --style_prompt "impressionism, Claude_Monet" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"


python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/impressionism/300_1_mean_diff_s_only_pooled_0_7/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/impressionism/300_1_mean_diff_s_only_pooled_0_7/" \
                       --style_prompt "impressionism, Claude_Monet" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"


python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/impressionism/300_1_mean_diff_s_only_pooled_0_9/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/impressionism/300_1_mean_diff_s_only_pooled_0_9/" \
                       --style_prompt "impressionism, Claude_Monet" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"

python metrics/clip.py --origin_dir "experiments/flux_schnell/style/generated_images/cyberpunk/0_0/steered" \
                       --target_dirs "experiments/flux_schnell/style/generated_images/impressionism/300_1_mean_diff_s_only_pooled_0_12/steered" \
                       --save_dir "experiments/flux_schnell/style/generated_images/impressionism/300_1_mean_diff_s_only_pooled_0_12/" \
                       --style_prompt "impressionism, Claude_Monet" \
                       --image_prompts "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt"