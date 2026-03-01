# python -m pytorch_fid  /home/jovyan/konovalova/steering/experiments/flux_dev/remove/generated_images_big_dataset/coco/1000_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_2/steered /home/jovyan/konovalova/steering/coco_real_images_correct_1024/

# python -m pytorch_fid  /home/jovyan/konovalova/steering/coco_real_images_correct_1024/ /home/jovyan/konovalova/steering/generated_images

pip install clean-fid
python -m clean_fid.fid --path1  /home/jovyan/konovalova/steering/coco_real_images_correct_1024/ --path2 /home/jovyan/konovalova/steering/generated_images