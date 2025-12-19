        
# python get_vector.py \
#         --task "style" \
#         --num_prompts 20 \
#         --pos_concept "sketches, pencil_drawing" \
#         --neg_concept "" \
#         --save_dir "style_experiments_flux_schnell_test/data_vectors" \
#         --exp_type "sketch_txt" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \


        
# python get_vector.py \
#         --task "style" \
#         --num_prompts 20 \
#         --pos_concept "anime_style, large_expressive_eyes" \
#         --neg_concept "" \
#         --save_dir "style_experiments_flux_schnell_test/data_vectors" \
#         --exp_type "anime_txt" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \


        
python get_vector.py \
        --task "style" \
        --num_prompts 20 \
        --pos_concept "sketches, pencil_drawing" \
        --neg_concept "" \
        --save_dir "style_experiments_flux_dev_test/data_vectors" \
        --exp_type "sketch" \
        --model_name "black-forest-labs/FLUX.1-dev" \
        --gs 3.5 \
        --num_inference_steps 20 \

