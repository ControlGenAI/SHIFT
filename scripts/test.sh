CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "remove" \
    --strength 0.0 \
    --strength_txt 10.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "results_debug/remove_sketch_normed_txt_100" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --remove_prompt "with sketches, pencil drawing style" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "remove" \
    --strength 10000.0 \
    --strength_txt 0.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "results_debug/remove_sketch" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --remove_prompt "with sketches, pencil drawing style" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "add concept" \
    --strength 1000.0 \
    --strength_txt 0.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "results_debug/add_sketch" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --remove_prompt "with sketches, pencil drawing style" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --data_dir "experiments/flux_schnell/style/final_steering/sketch" \
    --task "add concept" \
    --strength 10000.0 \
    --strength_txt 0.0 \
    --top_k_percent 0.01 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
    --results_dir "results_debug/add_sketch" \
    --inference_steps 4 \
    --seed 42 \
    --steer_txt \
    --vector_type 'diff' \
    --steering_type 'separate' \
    --guidance_scale 0.0 \
    --use_cls \
    --remove_prompt "with sketches, pencil drawing style" \
    --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 

# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 0.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "results_debug/remove_sketch" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 


# CUDA_VISIBLE_DEVICES=2,3, python apply_steering_with_injection_flux.py \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --data_dir "experiments/flux_schnell/style/final_steering/anime" \
#     --task "remove" \
#     --strength 100.0 \
#     --strength_txt 1.0 \
#     --top_k_percent 0.01 \
#     --min_signal_threshold 0.05 \
#     --cls_min 20.0 \
#     --results_dir "results_debug/remove_sketch" \
#     --inference_steps 4 \
#     --seed 42 \
#     --steer_txt \
#     --vector_type 'diff' \
#     --steering_type 'separate' \
#     --guidance_scale 0.0 \
#     --use_cls \
#     --remove_prompt "with sketches, pencil drawing style" \
#     --prompts_path "data/coco_captions_2017/coco_val2017_subset_250_seed42.txt" 