"""
Optimal Transport based steering vector calculation.

GPU-accelerated (PyTorch) OT solvers with log-domain Sinkhorn for stability.
CLI matches the original pipeline: timesteps/blocks, activation vs text, and
save naming compatible with apply_steering_with_injection_flux.py
(`base_{threshold}_{n_samples}_ot_sw.pt`, `..._text_ot_sw.pt`).
"""

import argparse
import os
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# GPU-accelerated OT solvers
# ---------------------------------------------------------------------------


class SinkhornOT:
    """Log-domain Sinkhorn for uniform marginals; supports n != m."""

    def __init__(self, epsilon: float = 0.05, max_iter: int = 100, tol: float = 1e-4):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol

    def compute_displacement(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Map source X (n, d) toward target Y (m, d) via entropic OT barycentric projection.
        Returns displacement (n, d): T(X) - X.
        """
        n, m = X.shape[0], Y.shape[0]
        device, dtype = X.device, X.dtype

        scale = torch.mean(torch.norm(X, dim=1)) + 1e-8
        X_n = X / scale
        Y_n = Y / scale

        C = torch.cdist(X_n, Y_n, p=2).pow(2)
        log_K = -C / self.epsilon

        # torch.full(..., tensor) ignores device and leaves vectors on CPU — build on X.device explicitly
        log_mu = torch.log(torch.ones(n, device=device, dtype=dtype) / n)
        log_nu = torch.log(torch.ones(m, device=device, dtype=dtype) / m)

        f = torch.zeros(n, device=device, dtype=dtype)
        g = torch.zeros(m, device=device, dtype=dtype)

        for _ in range(self.max_iter):
            f_prev = f.clone()
            f = log_mu - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
            g = log_nu - torch.logsumexp(log_K + f.unsqueeze(1), dim=0)
            if (f - f_prev).abs().max() < self.tol:
                break

        log_P = f.unsqueeze(1) + g.unsqueeze(0) + log_K
        P = torch.exp(log_P)
        row_sum = P.sum(dim=1, keepdim=True).clamp(min=1e-16)
        T_scaled = (P / row_sum) @ Y_n
        T_X = T_scaled * scale
        return T_X - X


class SlicedWassersteinOT:
    """Sliced Wasserstein displacement; X and Y must have the same shape (n, d)."""

    def __init__(self, n_projections: int = 1000, seed: int = 42):
        self.n_projections = n_projections
        self.seed = seed

    def compute_displacement(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        n, d = X.shape
        assert X.shape == Y.shape, f"SW requires matching shapes: {X.shape} vs {Y.shape}"
        device = X.device
        dtype = X.dtype

        g = torch.Generator(device=device)
        g.manual_seed(int(self.seed))
        dirs = torch.randn(self.n_projections, d, device=device, dtype=dtype, generator=g)
        dirs = dirs / torch.norm(dirs, dim=1, keepdim=True).clamp(min=1e-16)

        px = X @ dirs.T
        py = Y @ dirs.T

        val_x, idx_x = torch.sort(px, dim=0)
        val_y, _ = torch.sort(py, dim=0)
        delta = val_y - val_x

        disp_projected = torch.zeros_like(px)
        disp_projected.scatter_(0, idx_x, delta)

        return (disp_projected @ dirs) / self.n_projections


# ---------------------------------------------------------------------------
# Steering construction
# ---------------------------------------------------------------------------


def _make_solver(args, sample_index: int):
    if args.method == "sinkhorn":
        return SinkhornOT(epsilon=args.epsilon, max_iter=args.sinkhorn_iter)
    return SlicedWassersteinOT(
        n_projections=args.n_projections,
        seed=args.seed + sample_index,
    )


def compute_steering_activation(data_pos, data_neg, args, device: str) -> dict:
    """
    Output: {step: {f'layer_{b}': tensor(n_tokens, hidden_dim)}}
    Only steps in [0, timesteps) and layers layer_0 .. layer_{blocks-1}.
    """
    out = {}
    for step in range(args.timesteps):
        out[step] = {}
        if step not in data_pos or step not in data_neg:
            continue
        for block in range(args.blocks):
            layer = f"layer_{block}"
            if layer not in data_pos[step] or layer not in data_neg[step]:
                continue
            #print(data_pos[0]['layer_0']['txt'].shape)
            pos_tensor = data_pos[step][layer]['txt'].squeeze(1).to(device).float()
            neg_tensor = data_neg[step][layer]['txt'].squeeze(1).to(device).float()
            n_samples = min(pos_tensor.size(0), neg_tensor.size(0), args.n_samples)

            disps = []
            for i in range(n_samples):
                solver = _make_solver(args, i)
                disps.append(
                    solver.compute_displacement(neg_tensor[i], pos_tensor[i])
                )
            out[step][layer] = torch.stack(disps).mean(0).cpu()

            print(
                f"  step {step} {layer}: shape={tuple(out[step][layer].shape)}, "
                f"||disp||={out[step][layer].float().norm():.4f}"
            )

    return out


def compute_steering_text(data_pos, data_neg, args, device: str) -> dict:
    """
    OT on sequence embeddings per sample; pooled uses mean difference.
    Output: {'sequence': tensor, 'pooled': tensor (if present)}
    """
    pos_seq = data_pos["sequence"][: args.n_samples].to(device).float()
    neg_seq = data_neg["sequence"][: args.n_samples].to(device).float()
    n_samples = min(len(pos_seq), len(neg_seq))

    disps = []
    for i in range(n_samples):
        solver = _make_solver(args, i)
        disps.append(solver.compute_displacement(neg_seq[i], pos_seq[i]))

    result = {"sequence": torch.stack(disps).mean(0).cpu()}

    if "pooled" in data_pos and "pooled" in data_neg:
        result["pooled"] = (
            data_pos["pooled"][:n_samples] - data_neg["pooled"][:n_samples]
        ).mean(0).cpu()

    print(
        f"  text sequence: shape={tuple(result['sequence'].shape)}, "
        f"||disp||={result['sequence'].float().norm():.4f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="OT steering vectors (activation or text); matches pipeline file naming."
    )
    parser.add_argument("--pos_path", required=True)
    parser.add_argument("--neg_path", required=True)
    parser.add_argument("--save_dir", default=None, help="Directory for output .pt (preferred).")
    parser.add_argument(
        "--save_path",
        default=None,
        help="Full path to output .pt. If set, overrides save_dir naming.",
    )
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=19)
    parser.add_argument("--method", choices=["sw", "sinkhorn"], default="sw")
    parser.add_argument(
        "--data_type",
        choices=["activation", "text"],
        default="activation",
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--n_projections", type=int, default=1000)
    parser.add_argument("--sinkhorn_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.save_path is None and args.save_dir is None:
        raise SystemExit("Provide --save_dir or --save_path.")

    print("Loading datasets...")
    data_pos = torch.load(args.pos_path, map_location="cpu")
    data_neg = torch.load(args.neg_path, map_location="cpu")

    ot_tag = f"ot_{args.method}"
    if args.data_type == "text":
        print(f"\nComputing text OT ({args.method})...")
        result = compute_steering_text(data_pos, data_neg, args, device)
        if args.save_path:
            save_path = args.save_path
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            prefix = os.path.join(
                args.save_dir, f"base_{args.threshold}_{args.n_samples}"
            )
            save_path = f"{prefix}_text_{ot_tag}.pt"
    else:
        print(f"\nComputing activation OT ({args.method})...")
        result = compute_steering_activation(data_pos, data_neg, args, device)
        if args.save_path:
            save_path = args.save_path
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            prefix = os.path.join(
                args.save_dir, f"base_{args.threshold}_{args.n_samples}"
            )
            save_path = f"{prefix}_{ot_tag}.pt"

    torch.save(result, save_path)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
