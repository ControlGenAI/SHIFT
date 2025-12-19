# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 25 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob_back/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_back/base_remove_25_no_svm'


# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.5 \
#        --strength 100 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob/base \
#        --t_structure 11110 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_difference/base'

CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
       --structure 1.0 \
       --strength 400 \
       --n_samples 20 \
       --data_dir remove_experiments/final_steering/snoopy_noisy_2/base \
       --t_structure 11110 \
       --block_structure 0 \
       --save_svm \
       --inference_steps 20 \
       --block_steering 'all' \
       --task "remove" \
       --cls_min 50 \
       --results_dir 'remove_experiments/generated_images/snoopy_difference/base'


# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.5 \
#        --strength 200 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob/base \
#        --t_structure 11110 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_difference/base'

# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.5 \
#        --strength 100 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob/base \
#        --t_structure 11110 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_difference/base'






# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.2 \
#        --strength 25 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/sketch/base \
#        --t_structure 0 \
#        --block_structure 15 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/sketch/base_remove_25_1_not_cliped_structure_05_15'

# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 0.8 \
#        --strength 25 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/sketch/base \
#        --t_structure 0 \
#        --block_structure 15 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/sketch/base_remove_25_1_not_cliped_structure_08_15'


# CUDA_VISIBLE_DIVECES=3,2,1,0 python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 15 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/sketch/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/sketch/base_remove_15_1_not_cliped'



# CUDA_VISIBLE_DIVECES=3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob_back/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_back_mick/base_remove_20'


# CUDA_VISIBLE_DIVECES=3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 45 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/Spongebob_back/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/Spongebob_back_mick/base_remove_45'


# CUDA_VISIBLE_DIVECES=1, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 10 \
#        --n_samples 20 \
#        --data_dir remove_experiments/final_steering/snoopy/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --save_svm \
#        --inference_steps 20 \
#        --block_steering 'all' \
#        --task "remove" \
#        --results_dir 'remove_experiments/generated_images/snoopy/base_remove_10'
