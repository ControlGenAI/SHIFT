
NUM_PROMPTS=20
POS_CONCEPT_PROMPT='Kelly McKernan'
NEG_CONCEPT_PROMPT=''
POS_CONCEPT_KEY="kelly_prompts_20"  # Used for folder naming

SAVE_IMAGE_DIR="experiments/flux_dev/style/generated_images"
SAVE_BASE_DIR="experiments/flux_dev/style/data_vectors_3"
SAVE_TXT_BASE_DIR="test/data_vectors_txt"
PYTHON_SCRIPT="get_vector.py" 

# Create the specific save directory for this concept pair
CURRENT_SAVE_DIR="${SAVE_BASE_DIR}/"
mkdir -p "$CURRENT_SAVE_DIR"
mkdir -p "$SAVE_IMAGE_DIR"

echo "Starting extraction for: $POS_CONCEPT_KEY"

# # # # --- Execution ---
# CUDA_VISIBLE_DIVECES=1, python "$PYTHON_SCRIPT" \
#         --task "style" \
#         --num_prompts "$NUM_PROMPTS" \
#         --pos_concept "$POS_CONCEPT_PROMPT" \
#         --neg_concept "$NEG_CONCEPT_PROMPT" \
#         --save_dir "$CURRENT_SAVE_DIR" \
#         --save_image_dir "$SAVE_IMAGE_DIR" \
#         --exp_type "$POS_CONCEPT_KEY" \
#         --model_name "black-forest-labs/FLUX.1-dev" \
#         --gs 3.5 \
#         --num_inference_steps 28 \
#         --batch_size 1 \
        

# echo "Done. Results saved in $CURRENT_SAVE_DIR"

# python get_encoding_vector.py \
#                     --pos_concept "Kelly McKernan" \
#                     --neg_concept "" \
#                     --num_prompts 20 \
#                     --model_name "black-forest-labs/FLUX.1-dev" \
#                     --save_dir "experiments/flux_dev/style/data_vectors_txt_3" \
#                     --task style


python calculate_steering_vectors.py \
   --neg_path "/home/jovyan/konovalova/steering/experiments/flux_dev/style/data_vectors_3/kelly_prompts_20__gs_3.5_prompts_20_neg_attn_enc_1.pt" \
   --pos_path "/home/jovyan/konovalova/steering/experiments/flux_dev/style/data_vectors_3/kelly_prompts_20__gs_3.5_prompts_20_pos_attn_enc_1.pt" \
   --method diff \
   --n_samples 20 \
   --timesteps 1 \
   --save_dir "experiments/flux_dev/remove/final_steering/kelly_prompts_20/"

python calculate_steering_vectors.py \
   --neg_path "/home/jovyan/konovalova/steering/experiments/flux_dev/style/data_vectors_3/kelly_prompts_20__gs_3.5_prompts_20_neg_attn_enc_1.pt" \
   --pos_path "/home/jovyan/konovalova/steering/experiments/flux_dev/style/data_vectors_3/kelly_prompts_20__gs_3.5_prompts_20_pos_attn_enc_1.pt" \
   --save_svm \
   --n_samples 20 \
   --timesteps 1 \
   --classifier 'none' \
   --save_dir "experiments/flux_dev/remove/final_steering/kelly_prompts_20/"

python calculate_steering_vectors.py \
   --neg_path "/home/jovyan/konovalova/steering/experiments/flux_dev/style/data_vectors_txt_3/style__Kelly McKernan_prompts_20_neg_embeddings.pt" \
   --pos_path "/home/jovyan/konovalova/steering/experiments/flux_dev/style/data_vectors_txt_3/style_Kelly McKernan_Kelly McKernan_prompts_20_pos_embeddings.pt" \
   --method text \
   --n_samples 20 \
   --save_dir "experiments/flux_dev/remove/final_steering/kelly_prompts_20/"

