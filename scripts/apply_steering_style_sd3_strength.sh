# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 100 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/anime/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering '0,1,2,3,4,5,6,7' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/anime/test_aci'

CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
       --structure 1.0 \
       --strength 10 \
       --n_samples 20 \
       --data_dir style_experiments/final_steering/anime/base \
       --t_structure 1700 \
       --block_structure 0 \
       --save_svm \
       --inference_steps 20 \
       --orthogonal_projection \
       --block_steering 'all' \
       --cls_min 1 \
       --results_dir 'style_experiments/generated_images/anime/base_svm_10_cls_min1'

CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
       --structure 1.0 \
       --strength 10 \
       --n_samples 20 \
       --data_dir style_experiments/final_steering/3d/base \
       --t_structure 1700 \
       --block_structure 0 \
       --save_svm \
       --inference_steps 20 \
       --orthogonal_projection \
       --block_steering 'all' \
       --cls_min 1 \
       --results_dir 'style_experiments/generated_images/3d/base_svm_10_cls_min1'

CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
       --structure 1.0 \
       --strength 10 \
       --n_samples 20 \
       --data_dir style_experiments/final_steering/impressionism/base \
       --t_structure 1700 \
       --block_structure 0 \
       --save_svm \
       --inference_steps 20 \
       --orthogonal_projection \
       --block_steering 'all' \
       --cls_min 1 \
       --results_dir 'style_experiments/generated_images/impressionism/base_svm_10_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/cyberpunk/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/cyberpunk/base_svm_20_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 10 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/sketch/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/sketch/base_svm_10_cls_min1'

# # CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
# #        --structure 1.0 \
# #        --strength 15 \
# #        --n_samples 20 \
# #        --data_dir style_experiments/final_steering/sketch/base \
# #        --t_structure 1700 \
# #        --block_structure 0 \
# #        --save_svm \
# #        --inference_steps 20 \
# #        --orthogonal_projection \
# #        --block_steering 'all' \
# #        --cls_min 1 \
# #        --results_dir 'style_experiments/generated_images/sketch/base_svm_15_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/sketch/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/sketch/base_svm_20_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 10 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/watercolor/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/watercolor/base_svm_10_cls_min1'

# # CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
# #        --structure 1.0 \
# #        --strength 15 \
# #        --n_samples 20 \
# #        --data_dir style_experiments/final_steering/watercolor/base \
# #        --t_structure 1700 \
# #        --block_structure 0 \
# #        --save_svm \
# #        --inference_steps 20 \
# #        --orthogonal_projection \
# #        --cls_min 1 \
# #        --block_steering 'all' \
# #        --results_dir 'style_experiments/generated_images/watercolor/base_svm_15_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/watercolor/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/watercolor/base_svm_20_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 10 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/photorealism/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/photorealism/base_svm_10_cls_min1'

# # CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
# #        --structure 1.0 \
# #        --strength 15 \
# #        --n_samples 20 \
# #        --data_dir style_experiments/final_steering/photorealism/base \
# #        --t_structure 1700 \
# #        --block_structure 0 \
# #        --save_svm \
# #        --inference_steps 20 \
# #        --orthogonal_projection \
# #        --block_steering 'all' \
# #        --cls_min 1 \
# #        --results_dir 'style_experiments/generated_images/photorealism/base_svm_15_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/photorealism/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/photorealism/base_svm_20_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 10 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/impressionism/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/impressionism/base_svm_10_cls_min1'

# # CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
# #        --structure 1.0 \
# #        --strength 15 \
# #        --n_samples 20 \
# #        --data_dir style_experiments/final_steering/impressionism/base \
# #        --t_structure 1700 \
# #        --block_structure 0 \
# #        --save_svm \
# #        --inference_steps 20 \
# #        --orthogonal_projection \
# #        --block_steering 'all' \
# #        --cls_min 1 \
# #        --results_dir 'style_experiments/generated_images/impressionism/base_svm_15_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/impressionism/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/impressionism/base_svm_20_cls_min1'


# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 10 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/3d/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/3d/base_svm_10_cls_min1'

# # CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
# #        --structure 1.0 \
# #        --strength 15 \
# #        --n_samples 20 \
# #        --data_dir style_experiments/final_steering/3d/base \
# #        --t_structure 1700 \
# #        --block_structure 0 \
# #        --save_svm \
# #        --inference_steps 20 \
# #        --orthogonal_projection \
# #        --block_steering 'all' \
# #        --cls_min 1 \
# #        --results_dir 'style_experiments/generated_images/3d/base_svm_15_cls_min1'

# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir style_experiments/final_steering/3d/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --orthogonal_projection \
#        --block_steering 'all' \
#        --cls_min 1 \
#        --results_dir 'style_experiments/generated_images/3d/base_svm_20_cls_min1'


