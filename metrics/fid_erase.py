import os
import shutil
import torch
import natsort
import pandas as pd
from cleanfid import fid  # Changed from pytorch_fid

# --- CONFIGURATION ---
DIR_PAIRS = [
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/0_0_42/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/1000_5_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_all_42/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/0_0_42/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_3_no_cos/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/100_6_mean_normals_s_one_vec_new_only_pooled_2_scaled_clip/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_0/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_04/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_09/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_10_19/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_one_vec_new_only_pooled_2_scaled_clip_no_cls/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_s_new_only_pooled_2_scaled_clip/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_diff_m_one_vec_new_only_pooled_2_scaled_clip/steered"
    # ),
    # (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_normals_s_one_vec_new_only_pooled_2_scaled_clip/steered"
    # # ),
    #     (
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/0_0/steered", 
    #     "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy/500_6_mean_normals_s_one_vec_new_only_pooled_2_scaled_clip/steered"
    # ),
]


# DIR_PAIRS = [
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_10",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_10"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_20",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_20"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_30",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_30"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_42",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_42"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_50",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_50"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_60",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_60"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_70",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_70"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_80",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_80"
#     ),
#     (
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy_origin/0_0_90",
#         "/home/jovyan/konovalova/erasing/notebooks/snoopy/0_0_90"
#     ),
    
# ]


DIR_PAIRS = [
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_10/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_10"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_20/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_20"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_30/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_30"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_42/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_42"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_50/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_50"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_60/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_60"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_70/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_70"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_80/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_80"
    ),
    (
        "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_90/steered", 
        "/home/jovyan/konovalova/minimalist_concept_erasure/baselines/ca/output_snoopy/images/ca_results_1_epochs/0_0_90"
    ),
    
]


# DIR_PAIRS = [
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_10/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_10/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_20/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_20/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_30/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_30/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_42/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_42/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_50/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_50/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_60/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_60/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_70/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_70/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_80/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_80/steered"
#     ),
#     (
#         "experiments/flux_schnell/remove/generated_images_big_ablation/snoopy_v100/0_0_90/steered", 
#         "/home/jovyan/konovalova/steering/experiments/flux_schnell/remove/generated_images_big_dataset/snoopy/500_6_90/steered"
#     ),
    
# ]




CONCEPTS = ["Pikachu"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_aggregated_fid(dir_pairs, concepts, device="cuda"):
    print("\n" + "="*50)
    print("STARTING AGGREGATED CLEAN-FID CALCULATION")
    print("="*50)

    # 1. Setup Temporary Directory Structure
    temp_root = "./temp_fid_aggregate"
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    
    for side in ["origin", "target"]:
        for concept in concepts:
            os.makedirs(os.path.join(temp_root, side, concept), exist_ok=True)

    # 2. Aggregate Images across all directory pairs
    print("Grouping images by concept...")
    ssss = 0
    jjjj = 0
    for orig_dir, target_dir in dir_pairs:
        # Origin side
        for fname in os.listdir(orig_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                for concept in concepts:
                    #print( concept.lower(), fname.lower())
                    if concept.lower() in fname.lower():
                       
                        unique_name = f"{os.path.basename(orig_dir)}_{ssss}_{fname}"
                        shutil.copy(os.path.join(orig_dir, fname), 
                                    os.path.join(temp_root, "origin", concept, unique_name))
                        
        ssss += 1
        
        # Target side
        for fname in os.listdir(target_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                for concept in concepts:
                    if concept.lower() in fname.lower():
                        unique_name = f"{os.path.basename(target_dir)}_{jjjj}_{fname}"
                        shutil.copy(os.path.join(target_dir, fname), 
                                    os.path.join(temp_root, "target", concept, unique_name))
                        
        jjjj += 1

    # 3. Calculate Clean-FID per Concept
    fid_results = []
    for concept in concepts:
        path_origin = os.path.join(temp_root, "origin", concept)
        path_target = os.path.join(temp_root, "target", concept)
        
        num_orig = len(os.listdir(path_origin))
        num_target = len(os.listdir(path_target))
        
        if num_orig > 1 and num_target > 1:
            print(f"\nCalculating Clean-FID for: {concept} ({num_orig} images)...")
            try:
                # clean-fid call
                score = fid.compute_fid(
                    path_origin, 
                    path_target, 
                    device=torch.device(device),
                    batch_size=32,
                   # num_workers=0 # Set to >0 if running on Linux for speed
                )
                fid_results.append({"Concept": concept, "FID": f"{score:.2f}"})
            except Exception as e:
                print(f"Error calculating FID for {concept}: {e}")
                fid_results.append({"Concept": concept, "FID": "Error"})
        else:
            print(f"Skipping {concept}: Not enough images (Orig: {num_orig}, Target: {num_target})")
            fid_results.append({"Concept": concept, "FID": "N/A"})

    # 4. Cleanup and Display
    shutil.rmtree(temp_root)
    
    fid_df = pd.DataFrame(fid_results)
    print("\n" + "="*50)
    print("AGGREGATED CLEAN-FID RESULTS")
    print("="*50)
    print(fid_df.to_string(index=False))
    return fid_df

if __name__ == "__main__":
    run_aggregated_fid(DIR_PAIRS, CONCEPTS, DEVICE)