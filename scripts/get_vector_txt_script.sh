# python get_encoding_vector.py \
#                     --pos_concept "sketches, pencil_drawing" \
#                     --neg_concept "" \
#                     --num_prompts 20 \
#                     --model_name "black-forest-labs/FLUX.1-schnell" \
#                     --save_dir "experiments/flux_schnell/style/data_vectors_txt" \
#                     --prompts_file "imagenet_classes.txt" \
#                     --task style


# # ================ flux-schnell style =============
python get_encoding_vector.py \
                    --pos_concept "anime style, large expressive eyes" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/style/data_vectors_txt" \
                    --prompts_file "imagenet_classes.txt" \
                    --task style \

python get_encoding_vector.py \
                    --pos_concept "sketches, pencil drawing" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/style/data_vectors_txt" \
                    --prompts_file "imagenet_classes.txt" \
                    --task style \


python get_encoding_vector.py \
                    --pos_concept "cyberpunk, neon lights" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/style/data_vectors_txt" \
                    --prompts_file "imagenet_classes.txt" \
                    --task style \


python get_encoding_vector.py \
                    --pos_concept "impressionism, Claude Monet" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/style/data_vectors_txt" \
                    --prompts_file "imagenet_classes.txt" \
                    --task style \


# ============ flux add ================
python get_encoding_vector.py \
                    --pos_concept "big apple" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --task 'concrete' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/add/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_add.txt" \

python get_encoding_vector.py \
                    --pos_concept "big hat" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --task 'concrete' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/add/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_add.txt" \

python get_encoding_vector.py \
                    --pos_concept "big smile" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --task 'concrete' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/add/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_add.txt" \

python get_encoding_vector.py \
                    --pos_concept "red lipstick" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --task 'concrete' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/add/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_add.txt" \

python get_encoding_vector.py \
                    --pos_concept "long beard" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --task 'concrete' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/add/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_add.txt" \

python get_encoding_vector.py \
                    --pos_concept "big glasses" \
                    --neg_concept "" \
                    --num_prompts 20 \
                    --task 'concrete' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/add/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_add.txt" \

# ============ flux switch ============

python get_encoding_vector.py \
                    --pos_concept "old" \
                    --neg_concept "young" \
                    --num_prompts 20 \
                    --task 'switch' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/switch/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_age.txt" \


python get_encoding_vector.py \
                    --pos_concept "woman" \
                    --neg_concept "man" \
                    --num_prompts 20 \
                    --task 'switch' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/switch/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_gender.txt" \

python get_encoding_vector.py \
                    --pos_concept "kettle" \
                    --neg_concept "vase" \
                    --num_prompts 20 \
                    --task 'switch' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/switch/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_switch.txt" \


python get_encoding_vector.py \
                    --pos_concept "cat" \
                    --neg_concept "dog" \
                    --num_prompts 20 \
                    --task 'switch' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/switch/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_switch.txt" \


python get_encoding_vector.py \
                    --pos_concept "apple" \
                    --neg_concept "banana" \
                    --num_prompts 20 \
                    --task 'switch' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/switch/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_switch.txt" \


python get_encoding_vector.py \
                    --pos_concept "car" \
                    --neg_concept "bicycle" \
                    --num_prompts 20 \
                    --task 'switch' \
                    --model_name "black-forest-labs/FLUX.1-schnell" \
                    --save_dir "experiments/flux_schnell/switch/data_vectors_txt" \
                    --prompts_file "prompts_collection/dataset_creation/dataset_prompts_switch.txt" \





