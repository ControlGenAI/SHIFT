"""
Subspace-AcT steering parameter extraction.

Output hierarchy (same as calculate_steering_vector_ot_2.py):
  {step: {layer_N: {"img": <payload>, "txt": <payload>}}}

--branch txt   : text stream only (default). Works with explicit "txt", raw tensors per layer, or one unlabeled tensor per dict layer.
--branch img   : image stream only (explicit "img", raw tensors, or one unlabeled tensor per dict layer).
--branch dual  : both streams (only if each layer has both "img" and "txt").

Each branch payload stores everything needed for inference-time Subspace-AcT:
  - method: "subspace_affine", "subspace_sinkhorn", or "subspace_meandiff"
  - U: concept subspace basis, shape (C, d)
  - mu_neg: neutral anchor in original space, shape (C,)
  - evr: explained variance ratio, shape (C,)
  - d: selected subspace dimension (may be d_sel+1 if --pca_delta_mode append_mean_dir)
  - affine: W (d, d) and b (d,) when pooled; with --cpca_tokenwise, per-token maps W (T, d, d), b (T, d), plus affine_per_token / affine_T
  - sinkhorn (z_pos, g, epsilon) OR meandiff (delta_z in z-space)

Contrastive PCA on paired differences (see ContrastivePCA / ContrastivePCATokenWise):
  - Default (pooled): mean-pool each sample over tokens → delta ∈ R^{N×C}, same as before.
  - --cpca_tokenwise: PCA on all paired token rows delta.reshape(N*T, C); inference uses Z = (A - mu_neg) U per token (A ∈ R^{T×C}).
  pca_delta_mode:
  - centered: PCA on delta - mean(delta) (default)
  - raw_delta: PCA on delta (mean shift can dominate early PCs)
  - append_mean_dir: PCA on centered delta, then append orthogonalized mean(delta) to U
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch

from subspace_act import (
    ContrastivePCA,
    ContrastivePCATokenWise,
    SinkhornTransport,
    AffineTransport,
)


def is_dual_stream(data: dict) -> bool:
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
    out = {}
    for step in data:
        if not isinstance(step, int):
            continue
        out[step] = {}
        for layer_key in data[step]:
            layer_data = data[step][layer_key]
            if isinstance(layer_data, dict) and branch in layer_data:
                t = layer_data[branch]
                while t.dim() > 3:
                    t = t.squeeze(1)
                out[step][layer_key] = t
    return out


def _stream_dict_nonempty(d: dict) -> bool:
    for step in d:
        if isinstance(step, int) and d.get(step):
            return True
    return False


def _squeeze_ntc(t: torch.Tensor) -> torch.Tensor:
    while t.dim() > 3:
        t = t.squeeze(1)
    return t


def _resolve_layer_tensor(val, stream: str) -> Optional[torch.Tensor]:
    """
    Map one layer's stored value to an (N, T, C) tensor for ``stream``.

    - Raw tensor: use as-is (single-stream file).
    - Dict with ``stream`` key: use that tensor.
    - Dict with both ``img`` and ``txt`` but missing ``stream``: cannot infer; return None.
    - Dict with only the *other* branch key: wrong stream; return None.
    - Dict with exactly one tensor child (any key): treat as unlabeled single stream      (e.g. text activations saved without a ``txt`` wrapper).
    """
    if torch.is_tensor(val):
        t = _squeeze_ntc(val)
        return t if t.dim() == 3 else None

    if not isinstance(val, dict):
        return None

    if stream in val and torch.is_tensor(val[stream]):
        t = _squeeze_ntc(val[stream])
        return t if t.dim() == 3 else None

    other = "txt" if stream == "img" else "img"
    if "img" in val and "txt" in val:
        return None

    if other in val and torch.is_tensor(val[other]) and stream not in val:
        return None

    tensor_children = [v for v in val.values() if torch.is_tensor(v)]
    if len(tensor_children) != 1:
        return None
    t = _squeeze_ntc(tensor_children[0])
    return t if t.dim() == 3 else None


def extract_stream_auto(data: dict, stream: str) -> dict:
    """
    Single-stream flexible loader: flat tensors, explicit branch keys, or one unlabeled tensor per dict layer.
    """
    out = {}
    for step in data:
        if not isinstance(step, int):
            continue
        out[step] = {}
        for layer_key in data[step]:
            t = _resolve_layer_tensor(data[step][layer_key], stream)
            if t is not None:
                out[step][layer_key] = t
    return out


def load_activation_stream(
    data_pos: dict,
    data_neg: dict,
    stream: str,
) -> Tuple[dict, dict]:
    """
    Load one stream from pos/neg files.

    Order:
      1) Branch dict with key ``stream`` (full or partial dual with explicit key).
      2) Auto: raw tensor per layer, or dict with exactly one tensor (unlabeled stream).
    """
    pos_b = extract_branch(data_pos, stream)
    neg_b = extract_branch(data_neg, stream)
    if _stream_dict_nonempty(pos_b):
        if not _stream_dict_nonempty(neg_b):
            raise SystemExit(
                f"neg_path is missing '{stream}' stream activations "
                "(branch dict empty after extraction)."
            )
        return pos_b, neg_b

    auto_pos = extract_stream_auto(data_pos, stream)
    auto_neg = extract_stream_auto(data_neg, stream)
    if _stream_dict_nonempty(auto_pos) and _stream_dict_nonempty(auto_neg):
        return auto_pos, auto_neg

    raise SystemExit(
        f"Could not load '{stream}' stream. Expected either:\n"
        f"  - dict['{stream}'] tensor per layer, or\n"
        f"  - tensor (N,T,C) per layer, or\n"
        f"  - dict with exactly one tensor value per layer (unlabeled single stream)."
    )


def _fit_subspace_payload(
    neg_acts: torch.Tensor,
    pos_acts: torch.Tensor,
    mode: str,
    subspace_dim: int,
    min_explained: float,
    sinkhorn_epsilon: float,
    sinkhorn_iter: int,
    pca_delta_mode: str = "centered",
    cpca_tokenwise: bool = False,
    meandiff_topk_tokens: int = 0,
    meandiff_neutral_k: int = 0,
) -> Dict[str, torch.Tensor]:
    CpcaCls = ContrastivePCATokenWise if cpca_tokenwise else ContrastivePCA
    cpca = CpcaCls(
        subspace_dim=subspace_dim, pca_delta_mode=pca_delta_mode
    ).fit(neg_acts, pos_acts)
    evr = cpca.explained_variance_ratio()
    if evr.numel() == 0:
        d_sel = 1
    else:
        d_sel = int((evr.cumsum(0) < min_explained).sum().item()) + 1
        d_sel = max(1, min(d_sel, cpca.U.shape[1]))

    # Refit with selected d so stored basis matches threshold.
    finalize_append = pca_delta_mode == "append_mean_dir"
    cpca = CpcaCls(
        subspace_dim=d_sel, pca_delta_mode=pca_delta_mode
    ).fit(neg_acts, pos_acts, finalize_append_mean=finalize_append)
    evr = cpca.explained_variance_ratio()
    d_stored = int(cpca.U.shape[1])

    if cpca_tokenwise:
        neg_f = neg_acts.float()
        pos_f = pos_acts.float()
        z_neg_ntd = cpca.project(neg_f)  # (N, T, d)
        z_pos_ntd = cpca.project(pos_f)
        neg_flat = neg_f.mean(0)
        pos_flat = pos_f.mean(0)
        z_neg = cpca.project(neg_flat)
        z_pos = cpca.project(pos_flat)
        z_neg_mean = z_neg_ntd.mean(dim=0)
        z_pos_mean = z_pos_ntd.mean(dim=0)
    else:
        neg_pool = neg_acts.float().mean(dim=1)
        pos_pool = pos_acts.float().mean(dim=1)
        z_neg = cpca.project(neg_pool)
        z_pos = cpca.project(pos_pool)
        z_neg_mean = z_neg.mean(dim=0)
        z_pos_mean = z_pos.mean(dim=0)

    payload: Dict[str, torch.Tensor] = {
        "U": cpca.U.cpu(),
        "mu_neg": cpca.mu_neg.cpu(),
        "evr": evr.cpu(),
        "d": torch.tensor(d_stored, dtype=torch.int64),
        "z_neg_mean": z_neg_mean.cpu(),
        "z_pos_mean": z_pos_mean.cpu(),
        "cpca_tokenwise": torch.tensor(bool(cpca_tokenwise)),
    }

    if mode == "subspace_affine":
        payload["method"] = "subspace_affine"
        if cpca_tokenwise:
            _, T_tok, _ = z_neg_ntd.shape
            Ws: List[torch.Tensor] = []
            bs: List[torch.Tensor] = []
            for t in range(T_tok):
                transport_t = AffineTransport().fit(
                    z_neg_ntd[:, t, :], z_pos_ntd[:, t, :]
                )
                Ws.append(transport_t.W)
                bs.append(transport_t.b)
            payload["W"] = torch.stack(Ws, dim=0).cpu()
            payload["b"] = torch.stack(bs, dim=0).cpu()
            payload["affine_per_token"] = torch.tensor(True)
            payload["affine_T"] = torch.tensor(int(T_tok), dtype=torch.int64)

            ############# sanity check:
            # Попарные расстояния между W_t и W_s (Frobenius)
            T = 512
            W_flat = payload["W"].reshape(T, -1)  # (T, d*d)
            W_pairwise = torch.cdist(W_flat, W_flat, p=2)  # (T, T)
            bs = payload["b"]
            # Попарные расстояния между b_t и b_s
            b_pairwise = torch.cdist(bs, bs, p=2)          # (T, T)

            # Комбинированная метрика (можно менять веса)
            alpha, beta = 1.0, 1.0
            pairwise_total = alpha * W_pairwise + beta * b_pairwise

            print("mean W distance:", W_pairwise.mean().item())
            print("mean b distance:", b_pairwise.mean().item())
            print("mean total distance:", pairwise_total.mean().item())
            print(payload["W"].shape, )
            
        else:
            transport = AffineTransport().fit(z_neg, z_pos)
            payload["W"] = transport.W.cpu()
            payload["b"] = transport.b.cpu()
            payload["affine_per_token"] = torch.tensor(False)
    elif mode == "subspace_sinkhorn":
        transport = SinkhornTransport(
            epsilon=sinkhorn_epsilon,
            max_iter=sinkhorn_iter,
        ).fit(z_neg, z_pos)
        payload["method"] = "subspace_sinkhorn"
        # Continuous barycentric extension uses only target points + dual potential
        # (memory O(m*d + m) instead of O(N*M) for full coupling).
        payload["z_pos"] = transport.z_pos.cpu()
        payload["g"] = transport.g.cpu()
        payload["epsilon"] = torch.tensor(float(sinkhorn_epsilon), dtype=torch.float32)
    elif mode == "subspace_meandiff":
        # Paired mean shift in subspace: T(z) = z + delta_z (inference blends with lambda).
        # Pooled:    delta_z has shape (d,) — one shift for all tokens.
        # Tokenwise: delta_z has shape (T, d) — per-position average contrastive shift.
        #            Requires T at inference to match T at extraction (true for txt=512
        #            always; for img only if height/width identical).
        payload["method"] = "subspace_meandiff"
        if cpca_tokenwise:
            delta_a_tok = (pos_acts.float() - neg_acts.float()).mean(dim=0)  # (T, C)
            
            if meandiff_topk_tokens > 0:
                assert False
                # Importance by average ||(A_pos - A_neg) U|| across paired samples.
                z_delta = (pos_acts.float() - neg_acts.float()) @ cpca.U  # (N, T, d)
                mean_norm = z_delta.norm(dim=-1).mean(dim=0)  # (T,)
                t = int(mean_norm.shape[0])
                k = min(int(meandiff_topk_tokens), t)
                top_tokens = mean_norm.topk(k).indices
                mask = torch.zeros(t, dtype=torch.bool, device=delta_a_tok.device)
                mask[top_tokens] = True
                delta_a_tok[~mask] = 0.0
                payload["delta_z_mask"] = mask.cpu()
                payload["delta_z_topk"] = torch.tensor(k, dtype=torch.int64)
            
            delta_z_tok = (delta_a_tok @ cpca.U)
            
            if meandiff_neutral_k > 0:
                
                delta_z_tok = delta_z_tok @ cpca.U.T
                # Remove component of tokenwise shift that lies in neutral (neg) PCA subspace.
                # Build neutral basis from all neg token rows: (N*T, C).
                neg_flat = neg_acts.float().mean(dim=0).reshape(-1, neg_acts.shape[-1])
                print(neg_flat.shape)
                neg_centered = neg_flat - neg_flat.mean(dim=0, keepdim=True)
                _, _, v_t = torch.linalg.svd(neg_centered, full_matrices=False)
                k_neu = min(int(meandiff_neutral_k), int(v_t.shape[0]))
                if k_neu > 0:
                    u_neutral = v_t[:k_neu].T  # (C, k_neu)
                    delta_z_tok = delta_z_tok - (delta_z_tok @ u_neutral) @ u_neutral.T
                    payload["delta_neutral_k"] = torch.tensor(k_neu, dtype=torch.int64)
                    delta_z_tok = delta_z_tok @ cpca.U
                                               # (T, d)
            payload["delta_z"] = delta_z_tok.cpu()
            payload["delta_z_tokenwise"] = torch.tensor(True)
            payload["delta_z_T"] = torch.tensor(int(delta_z_tok.shape[0]), dtype=torch.int64)
        else:
            payload["delta_z"] = (z_pos - z_neg).mean(dim=0).cpu()
            payload["delta_z_tokenwise"] = torch.tensor(False)
    else:
        raise ValueError(f"Unsupported Subspace-AcT mode: {mode}")

    return payload


def compute_branch_subspace(
    pos: dict,
    neg: dict,
    *,
    timesteps: int,
    blocks: int,
    n_samples: int,
    mode: str,
    subspace_dim: int,
    min_explained: float,
    sinkhorn_epsilon: float,
    sinkhorn_iter: int,
    pca_delta_mode: str = "centered",
    cpca_tokenwise: bool = False,
    meandiff_topk_tokens: int = 0,
    meandiff_neutral_k: int = 0,
) -> dict:
    out = {}
    for step in range(timesteps):
        out[step] = {}
        if step not in pos or step not in neg:
            continue
        for block in range(blocks):
            layer = f"layer_{block}"
            if layer not in pos[step] or layer not in neg[step]:
                continue

            pos_t = pos[step][layer].float()
            neg_t = neg[step][layer].float()
            n = min(pos_t.size(0), neg_t.size(0), n_samples)
            if n < 2:
                continue

            payload = _fit_subspace_payload(
                neg_t[:n],
                pos_t[:n],
                mode=mode,
                subspace_dim=subspace_dim,
                min_explained=min_explained,
                sinkhorn_epsilon=sinkhorn_epsilon,
                sinkhorn_iter=sinkhorn_iter,
                pca_delta_mode=pca_delta_mode,
                cpca_tokenwise=cpca_tokenwise,
                meandiff_topk_tokens=meandiff_topk_tokens,
                meandiff_neutral_k=meandiff_neutral_k,
            )
            out[step][layer] = payload
            d_i = int(payload["d"])
            tw = "tokenwise" if cpca_tokenwise else "pooled"
            print(
                f"  step {step} {layer} [{mode}] cpca={tw} pca={pca_delta_mode} d={d_i} "
                f"EVR@d={payload['evr'][:d_i].sum().item():.4f}"
            )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Subspace-AcT extraction from transformer activations. "
        "Default: text stream only (--branch txt). Use --branch dual only if each layer has both img and txt."
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
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument(
        "--data_type",
        choices=["activation", "text", "dual"],
        default="activation",
        help="Legacy OT2 flag; only 'text' is rejected. Single-stream .pt files: use activation (default).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        choices=["dual", "img", "txt"],
        default="txt",
        help="Which stream to fit: txt (default), img, or dual (requires img+txt per layer).",
    )

    # Keep old names but use Subspace-AcT values.
    parser.add_argument(
        "--img_mode",
        choices=["subspace_affine", "subspace_sinkhorn", "subspace_meandiff"],
        default="subspace_affine",
    )
    parser.add_argument(
        "--txt_mode",
        choices=["subspace_affine", "subspace_sinkhorn", "subspace_meandiff"],
        default="subspace_affine",
    )
    parser.add_argument("--subspace_dim", type=int, default=32)
    parser.add_argument("--min_explained", type=float, default=0.90)
    parser.add_argument(
        "--pca_delta_mode",
        type=str,
        default="centered",
        choices=["centered", "raw_delta", "append_mean_dir"],
        help="How to build PCA rows from paired deltas: centered (default), "
        "raw_delta (no row centering), or append_mean_dir (centered PCA + μ_δ orth to U). "
        "With --cpca_tokenwise, rows are all (N*T) token differences; otherwise N pooled rows.",
    )
    parser.add_argument(
        "--cpca_tokenwise",
        action="store_true",
        help="Use ContrastivePCATokenWise: PCA on per-token paired differences; "
        "fit Affine/Sinkhorn/meandiff on z for every token row. Saved payloads set cpca_tokenwise.",
    )
    parser.add_argument("--sinkhorn_iter", type=int, default=100)
    parser.add_argument(
        "--meandiff_topk_tokens",
        type=int,
        default=0,
        help="If > 0 and mode=subspace_meandiff with --cpca_tokenwise, keep shift only for top-k important tokens.",
    )
    parser.add_argument(
        "--meandiff_neutral_k",
        type=int,
        default=0,
        help="If > 0 and mode=subspace_meandiff with --cpca_tokenwise, subtract projection of tokenwise shift onto top-k neutral PCs from neg activations.",
    )
    parser.add_argument("--max_tokens_sinkhorn", type=int, default=1024)
    parser.add_argument("--n_projections", type=int, default=1000)
    parser.add_argument("--use_partial_ot", action="store_true")
    parser.add_argument("--concept_percentile", type=float, default=0.7)
    parser.add_argument("--normalize_for_ot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.save_path is None and args.save_dir is None:
        raise SystemExit("Provide --save_dir or --save_path.")

    data_pos = torch.load(args.pos_path, map_location="cpu")
    data_neg = torch.load(args.neg_path, map_location="cpu")

    if not isinstance(data_pos, dict) or not isinstance(data_neg, dict):
        raise SystemExit("pos_path and neg_path must load to dict-like activation archives.")

    if args.data_type == "text":
        raise SystemExit("data_type=text is not supported for Subspace-AcT activations; use branch img/txt/dual on .pt activations.")
    if args.meandiff_topk_tokens < 0:
        raise SystemExit("--meandiff_topk_tokens must be >= 0.")
    if args.meandiff_neutral_k < 0:
        raise SystemExit("--meandiff_neutral_k must be >= 0.")
    if args.meandiff_topk_tokens > 0 and not args.cpca_tokenwise:
        print("Warning: --meandiff_topk_tokens is active only with --cpca_tokenwise; flag will be ignored.")
    if args.meandiff_neutral_k > 0 and not args.cpca_tokenwise:
        print("Warning: --meandiff_neutral_k is active only with --cpca_tokenwise; flag will be ignored.")

    img_result = {}
    txt_result = {}

    if args.branch == "dual":
        if not is_dual_stream(data_pos) or not is_dual_stream(data_neg):
            raise SystemExit(
                "branch=dual requires full dual-stream files (both 'img' and 'txt' per layer). "
                "Use --branch img or --branch txt for single-stream data."
            )
        print(
            f"branch=dual: img={args.img_mode}, txt={args.txt_mode}, "
            f"pca_delta_mode={args.pca_delta_mode} cpca_tokenwise={args.cpca_tokenwise}"
        )
        img_pos = extract_branch(data_pos, "img")
        img_neg = extract_branch(data_neg, "img")
        txt_pos = extract_branch(data_pos, "txt")
        txt_neg = extract_branch(data_neg, "txt")
        img_result = compute_branch_subspace(
            img_pos,
            img_neg,
            timesteps=args.timesteps,
            blocks=args.blocks,
            n_samples=args.n_samples,
            mode=args.img_mode,
            subspace_dim=args.subspace_dim,
            min_explained=args.min_explained,
            sinkhorn_epsilon=args.epsilon,
            sinkhorn_iter=args.sinkhorn_iter,
            pca_delta_mode=args.pca_delta_mode,
            cpca_tokenwise=args.cpca_tokenwise,
            meandiff_topk_tokens=args.meandiff_topk_tokens,
            meandiff_neutral_k=args.meandiff_neutral_k,
        )
        txt_result = compute_branch_subspace(
            txt_pos,
            txt_neg,
            timesteps=args.timesteps,
            blocks=args.blocks,
            n_samples=args.n_samples,
            mode=args.txt_mode,
            subspace_dim=args.subspace_dim,
            min_explained=args.min_explained,
            sinkhorn_epsilon=args.epsilon,
            sinkhorn_iter=args.sinkhorn_iter,
            pca_delta_mode=args.pca_delta_mode,
            cpca_tokenwise=args.cpca_tokenwise,
            meandiff_topk_tokens=args.meandiff_topk_tokens,
            meandiff_neutral_k=args.meandiff_neutral_k,
        )
    elif args.branch == "img":
        print(
            f"branch=img: mode={args.img_mode}, pca_delta_mode={args.pca_delta_mode} "
            f"cpca_tokenwise={args.cpca_tokenwise}"
        )
        img_pos, img_neg = load_activation_stream(data_pos, data_neg, "img")
        img_result = compute_branch_subspace(
            img_pos,
            img_neg,
            timesteps=args.timesteps,
            blocks=args.blocks,
            n_samples=args.n_samples,
            mode=args.img_mode,
            subspace_dim=args.subspace_dim,
            min_explained=args.min_explained,
            sinkhorn_epsilon=args.epsilon,
            sinkhorn_iter=args.sinkhorn_iter,
            pca_delta_mode=args.pca_delta_mode,
            cpca_tokenwise=args.cpca_tokenwise,
            meandiff_topk_tokens=args.meandiff_topk_tokens,
            meandiff_neutral_k=args.meandiff_neutral_k,
        )
    else:
        print(
            f"branch=txt: mode={args.txt_mode}, pca_delta_mode={args.pca_delta_mode} "
            f"cpca_tokenwise={args.cpca_tokenwise}"
        )
        txt_pos, txt_neg = load_activation_stream(data_pos, data_neg, "txt")
        txt_result = compute_branch_subspace(
            txt_pos,
            txt_neg,
            timesteps=args.timesteps,
            blocks=args.blocks,
            n_samples=args.n_samples,
            mode=args.txt_mode,
            subspace_dim=args.subspace_dim,
            min_explained=args.min_explained,
            sinkhorn_epsilon=args.epsilon,
            sinkhorn_iter=args.sinkhorn_iter,
            pca_delta_mode=args.pca_delta_mode,
            cpca_tokenwise=args.cpca_tokenwise,
            meandiff_topk_tokens=args.meandiff_topk_tokens,
            meandiff_neutral_k=args.meandiff_neutral_k,
        )

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

    if isinstance(data_pos, dict) and "img_latent_hw" in data_pos:
        result["img_latent_hw"] = data_pos["img_latent_hw"]
        result["img_resolution"] = data_pos.get("img_resolution")
        print(f"Propagated latent dims: {result['img_latent_hw']}")

    pca_tag = f"pca-{args.pca_delta_mode}"
    if args.cpca_tokenwise:
        pca_tag = f"{pca_tag}_tokenwise"
    if args.branch == "dual":
        tag = f"dual_img-{args.img_mode}_txt-{args.txt_mode}_{pca_tag}_subspace_act_v1"
    elif args.branch == "img":
        tag = f"img-{args.img_mode}_{pca_tag}_subspace_act_v1"
    else:
        tag = f"txt-{args.txt_mode}_{pca_tag}_subspace_act_v1"
    if args.save_path:
        save_path = args.save_path
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        prefix = os.path.join(args.save_dir, f"base_{args.threshold}_{args.n_samples}")
        save_path = f"{prefix}_{tag}.pt"

    torch.save(result, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
