python ./src/steering/calculate_text_encoder_pooled_monge.py \
  --pos_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete_big glasses_big glasses_prompts_20_pos_embeddings.pt" \
  --neg_path "/home/jovyan/konovalova/steering/experiments/flux_schnell/add/data_vectors_txt/concrete__big glasses_prompts_20_neg_embeddings.pt" \
  --save_path "/home/jovyan/konovalova/clean_code/steering/experiments/flux_schnell/test_compare_add/final_steering/subspace_act_txt_meandiff_tokenwise_raw_delta_attn_tokenwise_T_top_meandiff_100/text_glasses.pt" \
  --n_samples 20 \
  --train_frac 1.0 \
  --subspace_dim 32 \
  --min_explained 0.90 \
  --pca_delta_mode raw_delta