        
python get_encoding_vector.py \
        --task "style" \
        --num_prompts 20 \
        --pos_concept "sketches, pencil_drawing" \
        --neg_concept "" \
        --save_dir "style_experiments_flux_schnell_test/data_vectors_txt" \
        --exp_type "sketch_txt" \
        --model_name "black-forest-labs/FLUX.1-schnell" \


        
python get_encoding_vector.py \
        --task "style" \
        --num_prompts 20 \
        --pos_concept "anime_style, large_expressive_eyes" \
        --neg_concept "" \
        --save_dir "style_experiments_flux_schnell_test/data_vectors_txt" \
        --exp_type "anime_txt" \
        --model_name "black-forest-labs/FLUX.1-schnell" \