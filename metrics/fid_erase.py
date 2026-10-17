"""Per-concept clean-FID between original and steered generations.

Origin/target directories are paired (e.g. one pair per seed); images of every
concept are pooled across all pairs before FID is computed.

Usage:
    python metrics/fid_erase.py \
        --origin_dirs <orig_seed_10> <orig_seed_20> ... \
        --target_dirs <steered_seed_10> <steered_seed_20> ... \
        --concepts Snoopy Mickey Spongebob Pikachu Dog Legislator
"""
import argparse
import os
import shutil
import tempfile
import torch
import pandas as pd
from cleanfid import fid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_aggregated_fid(dir_pairs, concepts, device="cuda"):
    print("\n" + "="*50)
    print("STARTING AGGREGATED CLEAN-FID CALCULATION")
    print("="*50)

    temp_root = tempfile.mkdtemp(prefix="temp_fid_aggregate_")

    for side in ["origin", "target"]:
        for concept in concepts:
            os.makedirs(os.path.join(temp_root, side, concept), exist_ok=True)

    print("Grouping images by concept...")
    for pair_idx, (orig_dir, target_dir) in enumerate(dir_pairs):
        for fname in os.listdir(orig_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                for concept in concepts:
                    if concept.lower() in fname.lower():
                        unique_name = f"{os.path.basename(orig_dir)}_{pair_idx}_{fname}"
                        shutil.copy(os.path.join(orig_dir, fname),
                                    os.path.join(temp_root, "origin", concept, unique_name))

        for fname in os.listdir(target_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                for concept in concepts:
                    if concept.lower() in fname.lower():
                        unique_name = f"{os.path.basename(target_dir)}_{pair_idx}_{fname}"
                        shutil.copy(os.path.join(target_dir, fname),
                                    os.path.join(temp_root, "target", concept, unique_name))

    fid_results = []
    for concept in concepts:
        path_origin = os.path.join(temp_root, "origin", concept)
        path_target = os.path.join(temp_root, "target", concept)

        num_orig = len(os.listdir(path_origin))
        num_target = len(os.listdir(path_target))

        if num_orig > 1 and num_target > 1:
            print(f"\nCalculating Clean-FID for: {concept} ({num_orig} images)...")
            try:
                score = fid.compute_fid(
                    path_origin,
                    path_target,
                    device=torch.device(device),
                    batch_size=32,
                )
                fid_results.append({"Concept": concept, "FID": f"{score:.2f}"})
            except Exception as e:
                print(f"Error calculating FID for {concept}: {e}")
                fid_results.append({"Concept": concept, "FID": "Error"})
        else:
            print(f"Skipping {concept}: Not enough images (Orig: {num_orig}, Target: {num_target})")
            fid_results.append({"Concept": concept, "FID": "N/A"})

    shutil.rmtree(temp_root)

    fid_df = pd.DataFrame(fid_results)
    print("\n" + "="*50)
    print("AGGREGATED CLEAN-FID RESULTS")
    print("="*50)
    print(fid_df.to_string(index=False))
    return fid_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-concept clean-FID between original and steered generations.")
    parser.add_argument("--origin_dirs", nargs="+", required=True, help="Unsteered generations (one dir per seed).")
    parser.add_argument("--target_dirs", nargs="+", required=True, help="Steered generations, same order as --origin_dirs.")
    parser.add_argument("--concepts", nargs="+", default=["Snoopy", "Mickey", "Spongebob", "Pikachu", "Dog", "Legislator"],
                        help="Concept names; matched case-insensitively against file names.")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    if len(args.origin_dirs) != len(args.target_dirs):
        parser.error("--origin_dirs and --target_dirs must have the same length")

    run_aggregated_fid(list(zip(args.origin_dirs, args.target_dirs)), args.concepts, args.device)
