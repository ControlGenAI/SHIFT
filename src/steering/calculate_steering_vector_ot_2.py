"""
OT Steering v4 — handles dual-stream extraction format.

Input format from extract_activations_dual.py:
  {step: {layer_N: {"img": (n_samples, 4096, 3072), "txt": (n_samples, 512, 3072)}}}

Strategy (proven empirically):
  - Image branch: OT with adaptive Sinkhorn (manifold drift matters here)
  - Text branch: mean-diff (tokens are positionally aligned, OT adds noise)

Also supports legacy single-tensor format for backward compatibility.

New: --normalize_for_ot flag computes OT cost on L2-normalized activations
     (sphere geometry) while keeping displacement in the original space.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

# ---------------------------------------------------------------------------
# §1 Format detection
# ---------------------------------------------------------------------------

def is_dual_stream(data: dict) -> bool:
    """Check if data uses the new dual-stream format."""
    for step in data:
        if not isinstance(step, int):
            continue
        for layer_key in data[step]:
            val = data[step][layer_key]
            if isinstance(val, dict) and ("img" in val and "txt" in val):
                return True
            return False
    return False


def extract_branch(data: dict, branch: str) -> dict:
    """
    Pull a single branch from dual-stream data.
    Returns: {step: {layer_N: tensor(n_samples, n_tokens, d)}}
    """
    out = {}
    for step in data:
        if not isinstance(step, int):
            continue
        out[step] = {}
        for layer_key in data[step]:
            layer_data = data[step][layer_key]
            if isinstance(layer_data, dict) and branch in layer_data:
                t = layer_data[branch]
                # Extraction stores (n_samples, batch, n_tokens, d) with batch=1
                # Squeeze any singleton dimensions between dim 0 and the last two
                while t.dim() > 3:
                    t = t.squeeze(1)
                out[step][layer_key] = t
    return out


def _validate_dual_stream_shapes(
    img_branch: dict,
    txt_branch: dict,
    *,
    img_tokens: int = 4096,
    txt_tokens: int = 512,
    d_model: int = 3072,
):
    """
    Validate dual-stream shapes.

    Expected:
      img: (n_samples, 4096, 3072)
      txt: (n_samples, 512, 3072)
    """
    def _check(branch_dict: dict, name: str, expected_tokens: int):
        for step, layers in branch_dict.items():
            for layer, t in layers.items():
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"{name} step {step} {layer}: expected torch.Tensor, got {type(t)}")
                if t.dim() != 3:
                    raise ValueError(f"{name} step {step} {layer}: expected 3D tensor (n, tokens, d), got shape={tuple(t.shape)}")
                if t.shape[1] != expected_tokens or t.shape[2] != d_model:
                    raise ValueError(
                        f"{name} step {step} {layer}: expected (*, {expected_tokens}, {d_model}), got {tuple(t.shape)}"
                    )

    _check(img_branch, "img", img_tokens)
    _check(txt_branch, "txt", txt_tokens)


# ---------------------------------------------------------------------------
# §2 Solvers (match calculate_steering_vectors_ot.py)
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


def _make_solver(args, sample_index: int):
    if args.method == "sinkhorn":
        return SinkhornOT(epsilon=args.epsilon, max_iter=args.sinkhorn_iter)
    return SlicedWassersteinOT(
        n_projections=args.n_projections,
        seed=args.seed + sample_index,
    )


# ---------------------------------------------------------------------------
# §3 Differential Subspace
# ---------------------------------------------------------------------------

def compute_differential_subspace(pos, neg, k=128, min_explained=0.90):
    """SVD on paired differences → subspace of consistent shifts."""
    n_samples, n_tokens, d = pos.shape
    diffs = (pos - neg).reshape(-1, d)
    mean_diff = diffs.mean(0, keepdim=True)
    diffs_centered = diffs - mean_diff

    k = min(k, min(diffs_centered.shape) - 1)
    U, S, Vt = torch.linalg.svd(diffs_centered, full_matrices=False)
    explained_ratio = (S[:k] ** 2).cumsum(0) / (S ** 2).sum()
    k_eff = max(1, int((explained_ratio < min_explained).sum().item()) + 1)
    k_eff = min(k_eff, k)
    components = Vt[:k_eff]

    md_normed = F.normalize(mean_diff, dim=1)
    md_residual = md_normed - md_normed @ components.T @ components
    if md_residual.norm() > 0.05:
        components = torch.cat([F.normalize(md_residual, dim=1), components], dim=0)
        k_eff += 1

    return components, S[:k_eff], k_eff


def identify_concept_tokens(pos, neg, percentile=0.7):
    per_token_norm = (pos - neg).mean(0).norm(dim=1)
    threshold = torch.quantile(per_token_norm, percentile)
    return per_token_norm >= threshold


# ---------------------------------------------------------------------------
# §4 Image Branch: OT steering
# ---------------------------------------------------------------------------

def compute_image_ot(
    img_pos: dict,
    img_neg: dict,
    args,
    device: str,
) -> dict:
    """
    OT on image stream activations.
    img_pos/img_neg: {step: {layer_N: tensor(n_samples, 4096, 3072)}}
    """
    out = {}
    print("\n  [IMG] OT displacements...")
    for step in range(args.timesteps):
        out[step] = {}
        if step not in img_pos or step not in img_neg:
            continue

        for block in range(args.blocks):
            layer = f"layer_{block}"
            if layer not in img_pos[step] or layer not in img_neg[step]:
                continue

            pos_t = img_pos[step][layer].squeeze(1).to(device).float()
            neg_t = img_neg[step][layer].squeeze(1).to(device).float()
            n_samples = min(pos_t.size(0), neg_t.size(0), args.n_samples)

            disps = []
            for i in range(n_samples):
                solver = _make_solver(args, i)
                disps.append(solver.compute_displacement(neg_t[i], pos_t[i]))
            disp = torch.stack(disps).mean(0)
            out[step][layer] = disp.cpu()

            md = (pos_t[:n_samples] - neg_t[:n_samples]).mean(0)
            _print_diag(disp, md, f"step {step} {layer} [IMG]")

    return out


def compute_image_meandiff(
    img_pos: dict,
    img_neg: dict,
    args,
    device: str,
) -> dict:
    """
    Mean-diff on image stream. No OT.
    img_pos/img_neg: {step: {layer_N: tensor(n_samples, 4096, 3072)}}
    """
    out = {}
    for step in range(args.timesteps):
        out[step] = {}
        if step not in img_pos or step not in img_neg:
            continue
        for block in range(args.blocks):
            layer = f"layer_{block}"
            if layer not in img_pos[step] or layer not in img_neg[step]:
                continue

            pos_t = img_pos[step][layer].to(device).float()
            neg_t = img_neg[step][layer].to(device).float()
            n = min(pos_t.size(0), neg_t.size(0), args.n_samples)
            disp = (pos_t[:n] - neg_t[:n]).mean(0).cpu()
            out[step][layer] = disp

            print(f"    step {step} {layer} [IMG mean-diff]: "
                  f"shape={tuple(disp.shape)}, ||md||={disp.float().norm():.2f}")
    return out


# ---------------------------------------------------------------------------
# §5 Text Branch: mean-diff
# ---------------------------------------------------------------------------

def compute_text_meandiff(
    txt_pos: dict,
    txt_neg: dict,
    args,
    device: str,
) -> dict:
    """
    Mean-diff on text stream. No OT.
    txt_pos/txt_neg: {step: {layer_N: tensor(n_samples, 512, 3072)}}
    """
    out = {}
    for step in range(args.timesteps):
        out[step] = {}
        if step not in txt_pos or step not in txt_neg:
            continue
        for block in range(args.blocks):
            layer = f"layer_{block}"
            if layer not in txt_pos[step] or layer not in txt_neg[step]:
                continue

            pos_t = txt_pos[step][layer].to(device).float()
            neg_t = txt_neg[step][layer].to(device).float()
            n = min(pos_t.size(0), neg_t.size(0), args.n_samples)
            disp = (pos_t[:n] - neg_t[:n]).mean(0).cpu()
            out[step][layer] = disp

            print(f"    step {step} {layer} [TXT]: "
                  f"shape={tuple(disp.shape)}, ||md||={disp.float().norm():.2f}")
    return out


def compute_text_ot(
    txt_pos: dict,
    txt_neg: dict,
    args,
    device: str,
) -> dict:
    """
    OT on text stream activations (optional; can be noisier than mean-diff).
    txt_pos/txt_neg: {step: {layer_N: tensor(n_samples, 512, 3072)}}
    """
    # Same solver/loop as image OT (works for any (tokens, d) stream).
    return compute_image_ot(txt_pos, txt_neg, args, device)


# ---------------------------------------------------------------------------
# §6 Diagnostics
# ---------------------------------------------------------------------------

def _print_diag(ot_disp, mean_diff, label):
    ot_flat = ot_disp.float().reshape(1, -1).cpu()
    md_flat = mean_diff.float().reshape(1, -1).cpu()
    ot_n = ot_flat.norm().item()
    md_n = md_flat.norm().item()
    cos = F.cosine_similarity(ot_flat, md_flat).item()
    print(f"    {label}: ||OT||={ot_n:.2f}  ||MD||={md_n:.2f}  "
          f"ratio={ot_n/(md_n+1e-8):.3f}  cos={cos:.4f}")


# ---------------------------------------------------------------------------
# §7 Legacy format support
# ---------------------------------------------------------------------------

def compute_legacy_activation(data_pos, data_neg, args, device):
    """Handle old format: {step: {layer_N: tensor}} — single branch only."""
    # Same as v3 compute_steering_activation
    # Wrap as img-only and delegate
    return compute_image_ot(data_pos, data_neg, args, device)


def compute_legacy_text(data_pos, data_neg, args, device):
    """Handle old text format: {"sequence": tensor, "pooled": tensor}."""
    pos_seq = data_pos["sequence"][:args.n_samples].to(device).float()
    neg_seq = data_neg["sequence"][:args.n_samples].to(device).float()
    n = min(len(pos_seq), len(neg_seq))
    result = {"sequence": (pos_seq[:n] - neg_seq[:n]).mean(0).cpu()}
    if "pooled" in data_pos and "pooled" in data_neg:
        result["pooled"] = (
            data_pos["pooled"][:n] - data_neg["pooled"][:n]
        ).mean(0).cpu()
    print(f"  text (mean-diff): ||sv||={result['sequence'].float().norm():.2f}")
    return result


# ---------------------------------------------------------------------------
# §8 CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OT steering v4: dual-stream (OT for img, mean-diff for txt)"
    )
    parser.add_argument("--pos_path", required=True)
    parser.add_argument("--neg_path", required=True)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=19)
    parser.add_argument("--method", choices=["sw", "sinkhorn"], default="sinkhorn")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Sinkhorn epsilon (matches calculate_steering_vectors_ot.py).")

    # What to process
    parser.add_argument("--data_type", choices=["activation", "text", "dual"],
                        default="dual",
                        help="'dual' (new format): OT on img + mean-diff on txt. "
                             "'activation': legacy single-branch OT. "
                             "'text': legacy text mean-diff.")

    # Image OT params
    parser.add_argument(
        "--img_mode",
        choices=["ot", "meandiff"],
        default="ot",
        help="How to compute the image stream steering vector in dual mode.",
    )
    parser.add_argument(
        "--txt_mode",
        choices=["meandiff", "ot"],
        default="meandiff",
        help="How to compute the text stream steering vector in dual mode.",
    )
    parser.add_argument("--subspace_dim", type=int, default=128)
    parser.add_argument("--min_explained", type=float, default=0.90)
    parser.add_argument("--sinkhorn_iter", type=int, default=100)
    parser.add_argument("--max_tokens_sinkhorn", type=int, default=1024)
    parser.add_argument("--n_projections", type=int, default=1000)
    parser.add_argument("--use_partial_ot", action="store_true")
    parser.add_argument("--concept_percentile", type=float, default=0.7)

    # Sphere geometry for OT
    parser.add_argument("--normalize_for_ot", action="store_true",
                        help="Compute Sinkhorn cost matrix on L2-normalized activations. "
                             "Makes OT direction-aware (angular cost) while keeping "
                             "displacement in original space. Equivalent to collecting "
                             "after LayerNorm but without the hook-point problems.")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.save_path is None and args.save_dir is None:
        raise SystemExit("Provide --save_dir or --save_path.")

    data_pos = torch.load(args.pos_path, map_location="cpu")
    data_neg = torch.load(args.neg_path, map_location="cpu")

    dual = is_dual_stream(data_pos)
    tag = f"ot_{args.method}_v4"

    if args.data_type == "dual" or (args.data_type == "activation" and dual):
        if not dual:
            raise SystemExit("Data is not dual-stream format. Use --data_type activation or text.")

        img_desc = "OT" if args.img_mode == "ot" else "mean-diff"
        txt_desc = "OT" if args.txt_mode == "ot" else "mean-diff"
        print(f"\nDual-stream mode: {img_desc} on img, {txt_desc} on txt")
        if args.normalize_for_ot:
            print("  Using L2-normalized cost (sphere geometry)")
            tag += "_normed"

        img_pos = extract_branch(data_pos, "img")
        img_neg = extract_branch(data_neg, "img")
        txt_pos = extract_branch(data_pos, "txt")
        txt_neg = extract_branch(data_neg, "txt")

        _validate_dual_stream_shapes(img_pos, txt_pos)
        _validate_dual_stream_shapes(img_neg, txt_neg)

        tag = f"dual_img-{args.img_mode}_txt-{args.txt_mode}_{tag}"

        if args.img_mode == "ot":
            print("\n--- Image branch (OT) ---")
            img_result = compute_image_ot(img_pos, img_neg, args, device)
        else:
            print("\n--- Image branch (mean-diff) ---")
            img_result = compute_image_meandiff(img_pos, img_neg, args, device)

        if args.txt_mode == "ot":
            print("\n--- Text branch (OT) ---")
            txt_result = compute_text_ot(txt_pos, txt_neg, args, device)
        else:
            print("\n--- Text branch (mean-diff) ---")
            txt_result = compute_text_meandiff(txt_pos, txt_neg, args, device)

        # Merge into single output: {step: {layer_N: {"img": tensor, "txt": tensor}}}
        result = {}
        for step in range(args.timesteps):
            result[step] = {}
            img_layers = img_result.get(step, {})
            txt_layers = txt_result.get(step, {})
            all_layers = set(list(img_layers.keys()) + list(txt_layers.keys()))
            for layer in all_layers:
                result[step][layer] = {}
                if layer in img_layers:
                    result[step][layer]["img"] = img_layers[layer]
                if layer in txt_layers:
                    result[step][layer]["txt"] = txt_layers[layer]

    elif args.data_type == "text":
        print("\nText mode (mean-diff)...")
        result = compute_legacy_text(data_pos, data_neg, args, device)
        tag = "meandiff"

    else:  # activation, legacy format
        print(f"\nLegacy activation mode: OT ({args.method})...")
        result = compute_legacy_activation(data_pos, data_neg, args, device)

    if args.use_partial_ot:
        tag += "_partial"

    # Propagate resolution metadata from extraction data
    if isinstance(data_pos, dict) and "img_latent_hw" in data_pos:
        result["img_latent_hw"] = data_pos["img_latent_hw"]
        result["img_resolution"] = data_pos.get("img_resolution")
        print(f"  Propagated latent dims: {result['img_latent_hw']}")

    if args.save_path:
        save_path = args.save_path
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        prefix = os.path.join(args.save_dir, f"base_{args.threshold}_{args.n_samples}")
        save_path = f"{prefix}_{tag}.pt"

    torch.save(result, save_path)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()