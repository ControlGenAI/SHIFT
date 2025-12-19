

# CUDA_VISIBLE_DIVECES=1,0, python apply_steering_with_injection_flux.py \
#        --structure 1.2 \
#        --strength 0.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_dev_test/final_steering/anime/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-dev \
#        --inference_steps 20 \
#        --save_svm \
#        --guidance_scale 3.5 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --results_dir 'style_experiments_flux_dev_test/generated_images/txt_blocks_test_anime_diff_all/test'



CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
       --structure 1.2 \
       --strength 300.0 \
       --n_samples 20 \
       --data_dir style_experiments_flux_dev_test/final_steering/sketch/base \
       --t_structure 1500 \
       --block_structure 0 \
       --block_steering 'all' \
       --model_name black-forest-labs/FLUX.1-dev \
       --inference_steps 20 \
       --save_svm \
       --guidance_scale 3.5 \
       --cls_min 21 \
       --block_steering 'all' \
       --results_dir 'style_experiments_flux_dev_test/generated_images/txt_blocks_test_sketch_diff_all/test'



CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
       --structure 1.2 \
       --strength 150.0 \
       --n_samples 20 \
       --data_dir style_experiments_flux_dev_test/final_steering/sketch/base \
       --t_structure 1500 \
       --block_structure 0 \
       --block_steering 'all' \
       --model_name black-forest-labs/FLUX.1-dev \
       --inference_steps 20 \
       --save_svm \
       --guidance_scale 3.5 \
       --cls_min 21 \
       --block_steering 'all' \
       --results_dir 'style_experiments_flux_dev_test/generated_images/txt_blocks_test_sketch_diff_all/test'


# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
#        --structure 1.2 \
#        --strength 0.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_schnell_test/final_steering/sketch_not_sparse/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-schnell \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 0.0 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --results_dir 'style_experiments_flux_schnell_test/generated_images/txt_blocks_test_sketch_all/test'



# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
#        --structure 1.2 \
#        --strength 300.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_schnell_test/final_steering/sketch_not_sparse/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-schnell \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 0.0 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --results_dir 'style_experiments_flux_schnell_test/generated_images/txt_blocks_test_sketch_diff_all/test'



# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
#        --structure 1.2 \
#        --strength 150.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_schnell_test/final_steering/sketch_not_sparse/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-schnell \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 0.0 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --results_dir 'style_experiments_flux_schnell_test/generated_images/txt_blocks_test_sketch_diff_all/test'


# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
#        --structure 0.0 \
#        --strength 150.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_schnell_test/final_steering/anime_not_sparse/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-schnell \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 0.0 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --results_dir 'style_experiments_flux_schnell_test/generated_images/txt_blocks_test_anime_diff/test'




# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 500.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_dev_test/final_steering/sketch/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-dev \
#        --inference_steps 20 \
#        --save_svm \
#        --guidance_scale 3.5 \
#        --cls_min 21 \
#        --block_steering '0,1,2,3,4,5,6' \
#        --results_dir 'style_experiments_flux_dev_test/generated_images/sketch/test_new_only_steered'



# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 100 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_dev_test/final_steering/sketch/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-dev \
#        --inference_steps 20 \
#        --save_svm \
#        --guidance_scale 3.5 \
#        --results_dir 'style_experiments_flux_dev_test/generated_images/anime/test_new_only_steered_01'


# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 20 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_dev_test/final_steering/anime/base \
#        --t_structure 1700 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-dev \
#        --inference_steps 20 \
#        --save_svm \
#        --guidance_scale 3.5 \
#        --results_dir 'style_experiments_flux_dev_test/generated_images/anime/test_new_only_steered_02'


# # CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
# #        --structure 1.0 \
# #        --strength 100 \
# #        --n_samples 20 \
# #        --data_dir style_experiments_flux_dev_test/final_steering/cyberpunk/base \
# #        --t_structure 1700 \
# #        --block_structure 0 \
# #        --block_steering 'all' \
# #        --model_name black-forest-labs/FLUX.1-dev \
# #        --inference_steps 20 \
# #        --save_svm \
# #        --guidance_scale 0.0 \
# #        --results_dir 'style_experiments_flux_dev_test/generated_images/cyberpunk/test_all_gs'


# # CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
# #        --structure 1.0 \
# #        --strength 50 \
# #        --n_samples 20 \
# #        --data_dir style_experiments_flux_dev_test/final_steering/cyberpunk/base \
# #        --t_structure 1700 \
# #        --block_structure 0 \
# #        --block_steering 'all' \
# #        --model_name black-forest-labs/FLUX.1-dev \
# #        --inference_steps 20 \
# #        --save_svm \
# #        --guidance_scale 0.0 \
# #        --results_dir 'style_experiments_flux_dev_test/generated_images/cyberpunk/test_all'
