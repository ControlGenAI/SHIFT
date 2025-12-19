# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
#        --structure 1.0 \
#        --strength 0.0 \
#        --n_samples 20 \
#        --data_dir remove_experiments_flux_schnell/final_steering/mickey/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-schnell \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 0.0 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --task 'remove' \
#        --results_dir 'remove_experiments_flux_schnell/generated_images/mickey_txt/test'

CUDA_VISIBLE_DIVECES=1,2,3, python apply_steering_with_injection_flux.py \
       --structure 0.5 \
       --strength 0.0 \
       --n_samples 20 \
       --data_dir remove_experiments_flux_schnell/final_steering/spongebob/base \
       --t_structure 1500 \
       --block_structure 0 \
       --block_steering 'all' \
       --model_name black-forest-labs/FLUX.1-schnell \
       --inference_steps 4 \
       --save_svm \
       --guidance_scale 0.0 \
       --cls_min 21 \
       --block_steering 'all' \
       --task 'remove' \
       --results_dir 'remove_experiments_flux_schnell/generated_images/test/test'

# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
#        --structure 0.5 \
#        --strength 0.0 \
#        --n_samples 20 \
#        --data_dir remove_experiments_flux_schnell/final_steering/spongebob/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-schnell \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 0.0 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --task 'remove' \
#        --results_dir 'remove_experiments_flux_schnell/generated_images/Spongebob_txt/test'





# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection_flux.py \
#        --structure 0.5 \
#        --strength 100.0 \
#        --n_samples 20 \
#        --data_dir remove_experiments_flux_schnell/final_steering/spongebob/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-schnell \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 0.0 \
#        --cls_min 21 \
#        --block_steering 'all' \
#        --task 'remove' \
#        --results_dir 'remove_experiments_flux_schnell/generated_images/spongebob_txt/test'

