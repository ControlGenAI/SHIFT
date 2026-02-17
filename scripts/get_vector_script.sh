#!/bin/bash
# ==================== flux-schnell switch ==================


# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='old'
# NEG_CONCEPT_PROMPT='young'
# POS_CONCEPT_KEY="age"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/switch/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/switch/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "switch" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_age.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='woman'
# NEG_CONCEPT_PROMPT='man'
# POS_CONCEPT_KEY="gender"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/switch/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/switch/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "switch" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_gender.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='kettle'
# NEG_CONCEPT_PROMPT='vase'
# POS_CONCEPT_KEY="kettle_vase"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/switch/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/switch/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "switch" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_switch.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='cat'
# NEG_CONCEPT_PROMPT='dog'
# POS_CONCEPT_KEY="cat_dog"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/switch/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/switch/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "switch" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_switch.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


# NUM_PROMPTS=40
# POS_CONCEPT_PROMPT='apple'
# NEG_CONCEPT_PROMPT='banana'
# POS_CONCEPT_KEY="apple_banana"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/switch/dataset_images_1"
# SAVE_BASE_DIR="experiments/flux_schnell/switch/data_vectors_image_not_mean"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "switch" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_switch.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"



# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='car'
# NEG_CONCEPT_PROMPT='bicycle'
# POS_CONCEPT_KEY="car_bicycle"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/switch/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/switch/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "switch" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_switch.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


# # ==================== flux-schnell add ==================

# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='big apple'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="apple"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/add/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/add/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_add.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='big hat'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="test"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/add/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/add/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_add.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='big smile'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="smile"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/add/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/add/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_add.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='red lipstick'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="lipstick"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/add/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/add/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_add.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='long beard'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="beard"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/add/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/add/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_add.txt

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


NUM_PROMPTS=50
POS_CONCEPT_PROMPT='big glasses'
NEG_CONCEPT_PROMPT=''
POS_CONCEPT_KEY="test"  # Used for folder naming

SAVE_IMAGE_DIR="experiments/flux_schnell/add/dataset_images"
SAVE_BASE_DIR="experiments/flux_schnell/add/test"
SAVE_TXT_BASE_DIR="test/data_vectors"
PYTHON_SCRIPT="get_vector.py" 

# Create the specific save directory for this concept pair
CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
mkdir -p "$CURRENT_SAVE_DIR"
mkdir -p "$SAVE_IMAGE_DIR"

echo "Starting extraction for: $POS_CONCEPT_KEY"

# --- Execution ---
CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
        --task "concrete" \
        --num_prompts "$NUM_PROMPTS" \
        --pos_concept "$POS_CONCEPT_PROMPT" \
        --neg_concept "$NEG_CONCEPT_PROMPT" \
        --save_dir "$CURRENT_SAVE_DIR" \
        --save_image_dir "$SAVE_IMAGE_DIR" \
        --exp_type "$POS_CONCEPT_KEY" \
        --model_name "black-forest-labs/FLUX.1-schnell" \
        --gs 0.0 \
        --num_inference_steps 4 \
        --batch_size 1 \
        --prompt_path prompts_collection/dataset_creation/dataset_prompts_add.txt

echo "Done. Results saved in $CURRENT_SAVE_DIR"


# # ==================== flux-schnell style ==================
# # --- Configuration ---
# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='sketches, pencil drawing'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="sketch"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/style/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/style/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


# # --- Configuration ---
# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='cyberpunk, neon lights'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="cyberpunk"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/style/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/style/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# # --- Configuration ---
# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='anime style, large expressive eyes'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="anime"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/style/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/style/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# # --- Configuration ---
# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='impressionism, Claude_Monet'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="impressionism"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_schnell/style/generated_images"
# SAVE_BASE_DIR="experiments/flux_schnell/style/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# #========================= flux dev =========================

# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='sketches, pencil drawing'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="sketch"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_dev/style/generated_images"
# SAVE_BASE_DIR="experiments/flux_dev/style/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-dev" \
#         --gs 3.5 \
#         --num_inference_steps 25 \
#         --batch_size 1

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


# # --- Configuration ---
# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='cyberpunk, neon lights'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="cyberpunk"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_dev/style/generated_images"
# SAVE_BASE_DIR="experiments/flux_dev/style/data_vectors/"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-dev" \
#         --gs 3.5 \
#         --num_inference_steps 25 \
#         --batch_size 1

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# # --- Configuration ---
# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='anime style, large expressive eyes'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="anime"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_dev/style/generated_images"
# SAVE_BASE_DIR="experiments/flux_dev/style/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-dev" \
#         --gs 3.5 \
#         --num_inference_steps 25 \
#         --batch_size 1

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# # --- Configuration ---
# NUM_PROMPTS=20
# POS_CONCEPT_PROMPT='impressionism, Claude_Monet'
# NEG_CONCEPT_PROMPT=''
# POS_CONCEPT_KEY="impressionism"  # Used for folder naming

# SAVE_IMAGE_DIR="experiments/flux_dev/style/generated_images"
# SAVE_BASE_DIR="experiments/flux_dev/style/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,3, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-dev" \
#         --gs 3.5 \
#         --num_inference_steps 25 \
#         --batch_size 1

# echo "Done. Results saved in $CURRENT_SAVE_DIR"


# ############ remove concept ####################


# NUM_PROMPTS=35
# POS_CONCEPT_PROMPT='Snoopy'
# POS_CONCEPT_KEY="Snoopy_big"  # Used for folder naming
# NEG_CONCEPT_PROMPT=''

# SAVE_IMAGE_DIR="experiments/flux_schnell/remove/dataset_images"
# SAVE_BASE_DIR="experiments/flux_schnell/remove/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,0, python "$PYTHON_SCRIPT" \
#         --task "concrete" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_remove_3.txt


# echo "Done. Results saved in $CURRENT_SAVE_DIR"



############ remove concept ####################


# NUM_PROMPTS=135
# POS_CONCEPT_PROMPT='Nudity'
# POS_CONCEPT_KEY="test"  # Used for folder naming
# NEG_CONCEPT_PROMPT=''

# SAVE_IMAGE_DIR="experiments/flux_schnell/remove/dataset_images"
# SAVE_BASE_DIR="experiments/flux_schnell/remove/data_vectors"
# SAVE_TXT_BASE_DIR="test/data_vectors_txt"
# PYTHON_SCRIPT="get_vector.py" 

# # Create the specific save directory for this concept pair
# CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
# mkdir -p "$CURRENT_SAVE_DIR"
# mkdir -p "$SAVE_IMAGE_DIR"

# echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1,2,0, python "$PYTHON_SCRIPT" \
#         --task "people" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-schnell" \
#         --gs 0.0 \
#         --num_inference_steps 4 \
#         --batch_size 1 \
#         --prompt_path prompts_collection/dataset_creation/dataset_prompts_remove_3.txt


# echo "Done. Results saved in $CURRENT_SAVE_DIR"