# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='Spongebob'
# SAVE_IMAGE_DIR='remove_experiments/test_images'
# SAVE_BASE_DIR='remove_experiments/data_vectors'

# PYTHON_SCRIPT='get_vector.py' 

# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/${POS_CONCEPT_KEY}"
# mkdir -p "$CURRENT_SAVE_DIR"

# python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "Spongebob" \
#         --num_inference_steps 20 \
#         --gs 4.5 \

NUM_PROMPTS=20
POS_CONCEPT_PROMPT='Spongebob'
SAVE_IMAGE_DIR='remove_experiments/test_images'
SAVE_BASE_DIR='remove_experiments/data_vectors'

PYTHON_SCRIPT='get_vector.py' 

CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/${POS_CONCEPT_KEY}"
mkdir -p "$CURRENT_SAVE_DIR"

python "$PYTHON_SCRIPT" \
        --task "concrete" \
        --num_prompts "$NUM_PROMPTS" \
        --pos_concept "$POS_CONCEPT_PROMPT" \
        --save_dir "$CURRENT_SAVE_DIR" \
        --save_image_dir "$SAVE_IMAGE_DIR" \
        --exp_type "Spongebob_1" \
        --num_inference_steps 20 \
        --gs 4.5 \



# python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "van gogh" \
#         --num_inference_steps 20 \
#         --gs 4.5 \