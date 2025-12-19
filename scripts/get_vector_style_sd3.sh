#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e





CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
       --structure 1.0 \
       --strength 250.0 \
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
       --results_dir 'style_experiments_flux_dev_test/generated_images/sketch/test_new_only_steered'




CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
       --structure 1.0 \
       --strength 500.0 \
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
       --block_steering '0,1,2,3,4,5,6' \
       --results_dir 'style_experiments_flux_dev_test/generated_images/sketch/test_new_only_steered'



CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
       --structure 1.0 \
       --strength 500.0 \
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
       --block_steering '0,1,2,3,4,5,6,7,8,9,10' \
       --results_dir 'style_experiments_flux_dev_test/generated_images/sketch/test_new_only_steered'




# --- 1. Configuration ---

# Ensure your Python script is named 'get_vector.py'
PYTHON_SCRIPT='get_vector.py' 
NUM_PROMPTS=20
NEG_CONCEPT="" # Use "" for the default negative concept (baseline)
SAVE_BASE_DIR='style_experiments_flux_dev_test/data_vectors'
SAVE_IMAGE_DIR='style_experiments_flux_dev_test/test_images'

# --- 2. Style Dictionary (Bash Associative Array) ---
# NOTE: Associative arrays require Bash 4.0 or newer.
declare -A STYLE_PROMPTS=(
    #["impressionism"]="impressionism, Claude_Monet"
    #["cyberpunk"]="cyberpunk, neon_lights"
    #["photorealism"]="photorealism, hyperrealism"
   # ["3d"]="3D_renders, computer_generated_images"
    ["sketch"]="sketches, pencil_drawing"
   # ["cartoons"]="kids_cartoons, animated_characters"
    #["watercolor"]="watercolor_style"
    #["anime"]="anime_style, large_expressive_eyes"

    # Note: 'none' is often excluded from a loop of positive styles, but can be added if needed.
)

# --- 3. Setup Directories ---
mkdir -p "$SAVE_BASE_DIR"
mkdir -p "$SAVE_IMAGE_DIR"

# --- 4. Main Loop ---

# Iterate over the keys (short style names) of the associative array
for POS_CONCEPT_KEY in "${!STYLE_PROMPTS[@]}"; do
    
    # Get the full prompt string corresponding to the key
    POS_CONCEPT_PROMPT="${STYLE_PROMPTS[$POS_CONCEPT_KEY]}"

    echo "=========================================================="
    echo "🚀 Starting experiment for Style: $POS_CONCEPT_KEY"
    echo "   Prompt: ${POS_CONCEPT_PROMPT:0:80}..." # Print first 80 chars
    echo "=========================================================="

    # Construct the unique save directory using the short key
    CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/${POS_CONCEPT_KEY}"
    mkdir -p "$CURRENT_SAVE_DIR"

    # Execute the Python script
    # We pass the full, long prompt string to --pos_concept argument.
    # We pass the short key to --save_dir, and Python should be adjusted 
    # to use the long string in the internal prompt function call.
    
    # IMPORTANT: Since your Python script automatically substitutes the key, 
    # we'll stick to passing the KEY and rely on the Python code's logic.
    # HOWEVER, based on your request, we will pass the full prompt. 
    # This requires your Python script to *NOT* perform the key-to-prompt substitution. 
    
    # We will pass the full PROMPT as --pos_concept, and the KEY as a separate argument for saving.
    # Since your original Python script doesn't take a separate "key" argument for saving, 
    # we'll pass the KEY for --pos_concept, which is what the Python code *expects* # for cleaner file naming, and assume you update your Python script to use the KEY 
    # for file naming and the FULL PROMPT for generation.
    
    # To run successfully with the last provided Python code:
    # 1. You MUST revert the changes in the Python script where it looks up the key.
    # 2. OR, you must ensure the Python code uses the KEY for file naming and the VALUE for prompting.
    
    # For simplicity, we will assume you use the KEY for the save directory and the KEY for the
    # Python script's --pos_concept, relying on the Python substitution logic:
    
    # python "$PYTHON_SCRIPT" \
    #     --task "style" \
    #     --num_prompts "$NUM_PROMPTS" \
    #     --pos_concept "$POS_CONCEPT_PROMPT" \
    #     --neg_concept "$NEG_CONCEPT" \
    #     --save_dir "$CURRENT_SAVE_DIR" \
    #     --save_image_dir "$SAVE_IMAGE_DIR" \
    #     --exp_type "$POS_CONCEPT_KEY" \
    #     --model_name "black-forest-labs/FLUX.1-schnell" \
    #     --num_inference_steps 4 \
    #     --gs 0.0 \


    python "$PYTHON_SCRIPT" \
        --task "style" \
        --num_prompts "$NUM_PROMPTS" \
        --pos_concept "$POS_CONCEPT_PROMPT" \
        --neg_concept "$NEG_CONCEPT" \
        --save_dir "$CURRENT_SAVE_DIR" \
        --save_image_dir "$SAVE_IMAGE_DIR" \
        --exp_type "$POS_CONCEPT_KEY" \
        --model_name "black-forest-labs/FLUX.1-dev" \
        --num_inference_steps 20 \
        --gs 3.5 \

    
        
    
    # If your Python script *did not* handle the substitution, you would use:
    # python "$PYTHON_SCRIPT" ... --pos_concept "$POS_CONCEPT_PROMPT" ...
    # But since your Python code was written to handle the key substitution and filename logic, 
    # passing the KEY is the correct way to interact with the *last provided Python code*.

    echo "✅ Finished $POS_CONCEPT_KEY. Vectors saved to: $CURRENT_SAVE_DIR"
    
done

echo "=========================================================="
echo "✨ All style vector extractions complete. ✨"
echo "=========================================================="

# python calculate_steering.py \
#        --pos_path "style_experiments_flux_dev_test/data_vectors/watercolor/watercolor__gs_3.5_prompts_20_pos_attn_enc.pt" \
#        --neg_path "style_experiments_flux_dev_test/data_vectors/watercolor/watercolor__gs_3.5_prompts_20_neg_attn_enc.pt" \
#        --n_samples 20 \
#        --save_dir style_experiments_flux_dev_test/final_steering/watercolor/base \
#        --save_svm \
#        --timesteps 20 \
#        --blocks 57 \



# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 500.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_dev_test/final_steering/watercolor/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-dev \
#        --inference_steps 20 \
#        --save_svm \
#        --guidance_scale 3.5 \
#        --cls_min 21 \
#        --results_dir 'style_experiments_flux_dev_test/generated_images/watercolor/test_new_only_steered'




# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 500.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_dev_test/final_steering/watercolor/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-dev \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 3.5 \
#        --cls_min 21 \
#        --block_steering '0,1,2,3,4,5,6' \
#        --results_dir 'style_experiments_flux_dev_test/generated_images/watercolor/test_new_only_steered'



# CUDA_VISIBLE_DIVECES=1,0,2,3, python apply_steering_with_injection.py \
#        --structure 1.0 \
#        --strength 500.0 \
#        --n_samples 20 \
#        --data_dir style_experiments_flux_dev_test/final_steering/watercolor/base \
#        --t_structure 1500 \
#        --block_structure 0 \
#        --block_steering 'all' \
#        --model_name black-forest-labs/FLUX.1-dev \
#        --inference_steps 4 \
#        --save_svm \
#        --guidance_scale 3.5 \
#        --cls_min 21 \
#        --block_steering '0,1,2,3,4,5,6,7,8,9,10' \
#        --results_dir 'style_experiments_flux_dev_test/generated_images/watercolor/test_new_only_steered'





