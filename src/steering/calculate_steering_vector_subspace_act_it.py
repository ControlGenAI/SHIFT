"""
Text steering vector calculator (pooled low-space / OT).

Separate utility aligned with calculate_steering_vector_subspace_act.py style,
but for text encoder pooled vectors.

Input .pt files are expected to be dicts with:
  - "pooled": (N, C) or (N, 1, C)
  - optional "sequence": (N, T, C)
"""

import argparse
import os
import sys
from pathlib import Path

import torch

STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))

from src.steering.calculate_text_encoder_pooled_monge import compute_pooled_stats


def main():
    parser = argparse.ArgumentParser(
        description="Calculate text pooled subspace/OT steering vectors."
    )
    parser.add_argument("--pos_path", required=True)
    parser.add_argument("--neg_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--name", type=str, default="base")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--subspace_dim", type=int, default=32)
    parser.add_argument("--min_explained", type=float, default=0.90)
    parser.add_argument(
        "--pca_delta_mode",
        type=str,
        default="raw_delta",
        choices=("centered", "raw_delta", "append_mean_dir"),
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--no_cv", action="store_true")
    parser.add_argument(
        "--save_mode",
        type=str,
        default="all",
        choices=("all", "subspace_mean", "monge"),
        help="What single-vector file(s) to save in addition to bundle stats.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    pos = torch.load(args.pos_path, map_location="cpu")
    neg = torch.load(args.neg_path, map_location="cpu")
    if not isinstance(pos, dict) or not isinstance(neg, dict):
        raise TypeError("pos_path/neg_path must be dict .pt with pooled/sequence keys.")

    stats = compute_pooled_stats(
        pos=pos,
        neg=neg,
        n_samples=args.n_samples,
        train_frac=args.train_frac,
        subspace_dim=args.subspace_dim,
        min_explained=args.min_explained,
        pca_delta_mode=args.pca_delta_mode,
        eps=args.eps,
        run_cv=not args.no_cv,
    )

    base = f"{args.name}_text_pooled_ot_subspace"
    bundle_path = os.path.join(args.save_dir, f"{base}_stats.pt")
    torch.save(stats, bundle_path)
    print(f"Saved bundle: {bundle_path}")

    if args.save_mode in ("all", "subspace_mean"):
        sub_path = os.path.join(args.save_dir, f"{base}_subspace_mean.pt")
        torch.save(stats["txt_steering_vector_subspace_mean"], sub_path)
        print(f"Saved vector: {sub_path}")

    if args.save_mode in ("all", "monge"):
        monge_path = os.path.join(args.save_dir, f"{base}_monge.pt")
        torch.save(stats["txt_steering_vector_monge"], monge_path)
        print(f"Saved vector: {monge_path}")


if __name__ == "__main__":
    main()

