

# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='big apple'
# SAVE_IMAGE_DIR='add_experiments_flux_schnell/test_images'
# SAVE_BASE_DIR='add_experiments_flux_schnell/data_vectors'

# PYTHON_SCRIPT='get_vector.py' 

# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/${POS_CONCEPT_KEY}"
# mkdir -p "$CURRENT_SAVE_DIR"

# python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "apple" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \

python get_encoding_vector.py \
        --num_prompts 20 \
        --pos_concept "fox" \
        --neg_concept "wolf" \
        --save_dir "add_experiments_flux_schnell/data_vectors_txt" \
        --exp_type "fox_wolf_txt" \
        --task "concrete" \
        --model_name "black-forest-labs/FLUX.1-schnell" \


NUM_PROMPTS=20
POS_CONCEPT_PROMPT='fox'
NEG_CONCEPT_PROMPT='wolf'
SAVE_IMAGE_DIR='add_experiments_flux_schnell/test_images'
SAVE_BASE_DIR='add_experiments_flux_schnell/data_vectors'

PYTHON_SCRIPT='get_vector.py' 

CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/${POS_CONCEPT_KEY}"
mkdir -p "$CURRENT_SAVE_DIR"

CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
        --task "concrete" \
        --num_prompts "$NUM_PROMPTS" \
        --pos_concept "$POS_CONCEPT_PROMPT" \
        --neg_concept "$NEG_CONCEPT_PROMPT" \
        --save_dir "$CURRENT_SAVE_DIR" \
        --save_image_dir "$SAVE_IMAGE_DIR" \
        --exp_type "fox_wold" \
        --model_name "black-forest-labs/FLUX.1-schnell" \
        --gs 0.0 \
        --num_inference_steps 4 \


# python get_encoding_vector.py \
#         --num_prompts 20 \
#         --pos_concept "cat" \
#         --neg_concept "dog" \
#         --save_dir "add_experiments_flux_schnell/data_vectors_txt" \
#         --exp_type "apple_txt" \
#         --task "concrete" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
