        
python get_encoding_vector_sd3.py \
        --num_prompts 20 \
        --pos_concept "Spongebob" \
        --neg_concept "" \
        --save_dir "remove_experiments/data_vectors_txt" \
        --exp_type "Spongebob_txt" \
        --task "concrete" \
        --model_name stabilityai/stable-diffusion-3.5-medium \

        
python get_encoding_vector_sd3.py \
        --num_prompts 20 \
        --pos_concept "snoopy" \
        --neg_concept "" \
        --save_dir "remove_experiments/data_vectors_txt" \
        --exp_type "snoopy_txt" \
        --task "concrete" \
        --model_name stabilityai/stable-diffusion-3.5-medium \
        
python get_encoding_vector_sd3.py \
        --num_prompts 20 \
        --pos_concept "sketches, pencil_drawing" \
        --neg_concept "" \
        --save_dir "style_experiments/data_vectors_txt" \
        --exp_type "sketch_txt" \
        --task "concrete" \
        --model_name stabilityai/stable-diffusion-3.5-medium \

        
python get_encoding_vector_sd3.py \
        --num_prompts 20 \
        --pos_concept "anime_style, large_expressive_eyes" \
        --neg_concept "" \
        --save_dir "style_experiments/data_vectors_txt" \
        --exp_type "anime_txt" \
        --task "style" \
        --model_name stabilityai/stable-diffusion-3.5-medium \
        
