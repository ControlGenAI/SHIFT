
NUM_PROMPTS=20
POS_CONCEPT_PROMPT='Mickey Mouse'
SAVE_IMAGE_DIR='remove_experiments_flux_schnell/test_images'
SAVE_BASE_DIR='remove_experiments_flux_schnell/data_vectors'

PYTHON_SCRIPT='get_vector.py' 

CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/${POS_CONCEPT_KEY}"
mkdir -p "$CURRENT_SAVE_DIR"

python "$PYTHON_SCRIPT" \
        --task "concrete" \
        --num_prompts "$NUM_PROMPTS" \
        --pos_concept "Spongebob" \
        --save_dir "$CURRENT_SAVE_DIR" \
        --save_image_dir "$SAVE_IMAGE_DIR" \
        --exp_type "Spongebob" \
        --model_name "black-forest-labs/FLUX.1-schnell" \
        --gs 0.0 \
        --num_inference_steps 4 \



# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='Snoopy'
# SAVE_IMAGE_DIR='remove_experiments_flux_schnell/test_images'
# SAVE_BASE_DIR='remove_experiments_flux_schnell/data_vectors'

# PYTHON_SCRIPT='get_vector.py' 

# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/${POS_CONCEPT_KEY}"
# mkdir -p "$CURRENT_SAVE_DIR"

# python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "Snoopy" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \



