"""
Считает mean-diff и Monge для текстового энкодера (формат как у calculate_steering_vectors.py --method text):
  входные .pt — dict с ключами 'pooled' (обязательно) и опционально 'sequence'.

  pooled: (N, C) или (N, 1, C)
  sequence: (N, T, C) — только mean-diff в полном пространстве по токенам

  Для pooled:
    - mean-diff в R^C (разность средних pos/neg)
    - ContrastivePCA → подпространство, mean-diff в R^d
    - коварианности z, матрица Monge M, сдвиги mu_neg_z, mu_pos_z
    - (опционально) CV: ||z_pred - z_pos|| для mean-diff vs Monge на val

  Сохраняет один .pt с полями, совместимыми с flux txt_steering / utils.apply_txt_steering_pooled_advanced.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.steering.subspace_act import ContrastivePCA


def _as_ntc_pooled(pooled: torch.Tensor) -> torch.Tensor:
    x = pooled.float()
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"pooled: ожидаю (N,C) или (N,1,C), got {tuple(x.shape)}")
    return x


def _covariance(z_nd: torch.Tensor, eps: float) -> torch.Tensor:
    zc = z_nd - z_nd.mean(dim=0, keepdim=True)
    n = max(1, z_nd.shape[0])
    cov = (zc.T @ zc) / n
    d = cov.shape[0]
    return cov + eps * torch.eye(d, device=cov.device, dtype=cov.dtype)


def _monge_matrix(Sigma_neg: torch.Tensor, Sigma_pos: torch.Tensor, eps: float) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(Sigma_neg)
    vals = vals.clamp(min=eps)
    s_sqrt = vecs @ torch.diag(vals.sqrt()) @ vecs.T
    s_invsqrt = vecs @ torch.diag(vals.rsqrt()) @ vecs.T
    mid = s_sqrt @ Sigma_pos @ s_sqrt
    vm, em = torch.linalg.eigh(mid)
    vm = vm.clamp(min=eps)
    mid_sqrt = em @ torch.diag(vm.sqrt()) @ em.T
    return s_invsqrt @ mid_sqrt @ s_invsqrt


def _fit_cpca_pooled(
    neg_ntc: torch.Tensor,
    pos_ntc: torch.Tensor,
    subspace_dim: int,
    min_explained: float,
    pca_delta_mode: str,
) -> ContrastivePCA:
    finalize_append = pca_delta_mode == "append_mean_dir"
    cpca = ContrastivePCA(
        subspace_dim=subspace_dim,
        pca_delta_mode=pca_delta_mode,
    ).fit(neg_ntc, pos_ntc)
    evr = cpca.explained_variance_ratio()
    if evr.numel() == 0:
        d_sel = 1
    else:
        d_sel = int((evr.cumsum(0) < min_explained).sum().item()) + 1
        d_sel = max(1, min(d_sel, int(cpca.U.shape[1])))
    return ContrastivePCA(
        subspace_dim=d_sel,
        pca_delta_mode=pca_delta_mode,
    ).fit(neg_ntc, pos_ntc, finalize_append_mean=finalize_append)


def _optional_cv(
    z_neg_tr: torch.Tensor,
    z_pos_tr: torch.Tensor,
    z_neg_va: torch.Tensor,
    z_pos_va: torch.Tensor,
    Sigma_neg: torch.Tensor,
    Sigma_pos: torch.Tensor,
    eps: float,
) -> Dict[str, float]:
    mu_neg_z = z_neg_tr.mean(dim=0)
    mu_pos_z = z_pos_tr.mean(dim=0)
    M = _monge_matrix(Sigma_neg, Sigma_pos, eps)
    err_mean = ((z_neg_va + (mu_pos_z - mu_neg_z)) - z_pos_va).norm(dim=1).mean().item()
    z_pred_m = mu_pos_z + (z_neg_va - mu_neg_z) @ M.T
    err_monge = (z_pred_m - z_pos_va).norm(dim=1).mean().item()
    return {
        "cv_mean_diff_z": err_mean,
        "cv_monge_z": err_monge,
        "cv_gain_z": err_mean - err_monge,
    }


def compute_pooled_stats(
    pos: Dict[str, Any],
    neg: Dict[str, Any],
    n_samples: Optional[int],
    train_frac: float,
    subspace_dim: int,
    min_explained: float,
    pca_delta_mode: str,
    eps: float,
    run_cv: bool,
) -> Dict[str, Any]:
    if "pooled" not in pos or "pooled" not in neg:
        raise KeyError("Ожидаю ключ 'pooled' в pos и neg (как в --method text).")

    pos_ntc = _as_ntc_pooled(pos["pooled"])
    neg_ntc = _as_ntc_pooled(neg["pooled"])
    if pos_ntc.shape != neg_ntc.shape:
        raise ValueError(f"pooled shape pos {tuple(pos_ntc.shape)} vs neg {tuple(neg_ntc.shape)}")

    n = pos_ntc.shape[0]
    if n_samples is not None:
        n = min(n, int(n_samples))
    pos_ntc = pos_ntc[:n].contiguous()
    neg_ntc = neg_ntc[:n].contiguous()

    h_pos = pos_ntc.mean(dim=1)
    h_neg = neg_ntc.mean(dim=1)
    mean_diff_c = h_pos.mean(dim=0) - h_neg.mean(dim=0)

    out: Dict[str, Any] = {
        "meta": {
            "n": n,
            "pooled_shape_ntc": tuple(pos_ntc.shape),
            "train_frac": train_frac,
            "subspace_dim_cap": subspace_dim,
            "min_explained": min_explained,
            "pca_delta_mode": pca_delta_mode,
            "eps": eps,
        },
        "pooled_mean_diff_c": mean_diff_c.cpu(),
    }

    if run_cv and n >= 6:
        n_train = max(4, int(n * train_frac))
        n_train = min(n_train, n - 2)
        pos_tr, pos_va = pos_ntc[:n_train], pos_ntc[n_train:]
        neg_tr, neg_va = neg_ntc[:n_train], neg_ntc[n_train:]
    else:
        pos_tr, pos_va = pos_ntc, pos_ntc[:0]
        neg_tr, neg_va = neg_ntc, neg_ntc[:0]
        n_train = n

    cpca = _fit_cpca_pooled(neg_tr, pos_tr, subspace_dim, min_explained, pca_delta_mode)
    U = cpca.U.float()
    mu_neg = cpca.mu_neg.float()
    d = U.shape[1]
    evr = cpca.explained_variance_ratio()

    z_neg_all = (h_neg - mu_neg) @ U
    z_pos_all = (h_pos - mu_neg) @ U
    mean_diff_z = mean_diff_c @ U

    out["pooled"] = {
        "cpca_d": d,
        "cpca_explained_variance_ratio": evr.cpu(),
        "mu_neg_c": mu_neg.cpu(),
        "V_d": U.cpu(),
        "mean_diff_z": mean_diff_z.cpu(),
        "z_neg_all": z_neg_all.cpu(),
        "z_pos_all": z_pos_all.cpu(),
    }

    z_neg_tr = (neg_tr.mean(dim=1) - mu_neg) @ U
    z_pos_tr = (pos_tr.mean(dim=1) - mu_neg) @ U
    mu_neg_z = z_neg_tr.mean(dim=0)
    mu_pos_z = z_pos_tr.mean(dim=0)
    Sigma_neg = _covariance(z_neg_tr, eps)
    Sigma_pos = _covariance(z_pos_tr, eps)
    M = _monge_matrix(Sigma_neg, Sigma_pos, eps)

    rel_shape = ((Sigma_pos - Sigma_neg).norm() / Sigma_neg.norm().clamp(min=eps)).item()
    out["pooled"]["Sigma_neg_z"] = Sigma_neg.cpu()
    out["pooled"]["Sigma_pos_z"] = Sigma_pos.cpu()
    out["pooled"]["M"] = M.cpu()
    out["pooled"]["mu_neg_z"] = mu_neg_z.cpu()
    out["pooled"]["mu_pos_z"] = mu_pos_z.cpu()
    out["pooled"]["relative_shape_diff_z"] = rel_shape

    z_neg_va = (neg_va.mean(dim=1) - mu_neg) @ U
    z_pos_va = (pos_va.mean(dim=1) - mu_neg) @ U
    if run_cv and z_neg_va.shape[0] > 0:
        out["pooled"]["cv"] = _optional_cv(
            z_neg_tr, z_pos_tr, z_neg_va, z_pos_va, Sigma_neg, Sigma_pos, eps
        )
        out["meta"]["n_train"] = n_train
        out["meta"]["n_val"] = n - n_train

    delta_z_m = (mu_pos_z + (z_neg_all - mu_neg_z) @ M.T) - z_neg_all
    monge_delta_c_per_sample = delta_z_m @ U.T
    out["pooled"]["monge_delta_z_per_sample_all"] = delta_z_m.cpu()
    out["pooled"]["monge_delta_c_per_sample_all"] = monge_delta_c_per_sample.cpu()
    out["pooled"]["monge_delta_c_mean_all"] = monge_delta_c_per_sample.mean(dim=0).cpu()

    mean_diff_c_in_span = U @ mean_diff_z
    out["pooled"]["mean_diff_c_in_subspace"] = mean_diff_c_in_span.cpu()

    out["txt_steering_vector_subspace_mean"] = {
        "pooled_steering_mode": "subspace_mean",
        "mu_neg": mu_neg.cpu(),
        "V_d": U.cpu(),
        "delta_z": mean_diff_z.cpu(),
    }
    out["txt_steering_vector_monge"] = {
        "pooled_steering_mode": "monge",
        "mu_neg": mu_neg.cpu(),
        "V_d": U.cpu(),
        "mu_neg_z": mu_neg_z.cpu(),
        "mu_pos_z": mu_pos_z.cpu(),
        "M": M.cpu(),
    }

    return out


def compute_sequence_mean_diff_only(
    pos: Dict[str, Any], neg: Dict[str, Any], n_samples: Optional[int]
) -> Optional[torch.Tensor]:
    if "sequence" not in pos or "sequence" not in neg:
        return None
    ps, ns = pos["sequence"].float(), neg["sequence"].float()
    if ps.shape != ns.shape:
        raise ValueError(f"sequence shape pos {tuple(ps.shape)} vs neg {tuple(ns.shape)}")
    n = ps.shape[0]
    if n_samples is not None:
        n = min(n, int(n_samples))
    ps, ns = ps[:n], ns[:n]
    return (ps - ns).mean(dim=0).cpu()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Mean-diff / low-d / Monge для текстового энкодера (pooled + опционально sequence)."
    )
    p.add_argument("--pos_path", type=str, required=True)
    p.add_argument("--neg_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True, help="Куда сохранить .pt (полный путь)")
    p.add_argument("--n_samples", type=int, default=None, help="Ограничить число пар (по умолчанию все)")
    p.add_argument("--train_frac", type=float, default=0.8, help="Доля train для CPCA и Monge (и CV)")
    p.add_argument("--subspace_dim", type=int, default=32)
    p.add_argument("--min_explained", type=float, default=0.90)
    p.add_argument("--pca_delta_mode", type=str, default="raw_delta", choices=("centered", "raw_delta", "append_mean_dir"))
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--no_cv", action="store_true", help="Не считать CV на holdout")
    args = p.parse_args()

    pos = torch.load(args.pos_path, map_location="cpu")
    neg = torch.load(args.neg_path, map_location="cpu")
    if not isinstance(pos, dict) or not isinstance(neg, dict):
        raise TypeError("pos_path / neg_path: ожидаю dict с ключами pooled/sequence.")

    out = compute_pooled_stats(
        pos,
        neg,
        n_samples=args.n_samples,
        train_frac=args.train_frac,
        subspace_dim=args.subspace_dim,
        min_explained=args.min_explained,
        pca_delta_mode=args.pca_delta_mode,
        eps=args.eps,
        run_cv=not args.no_cv,
    )

    seq_md = compute_sequence_mean_diff_only(pos, neg, args.n_samples)
    if seq_md is not None:
        out["sequence_mean_diff_tc"] = seq_md

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True)
    torch.save(out, args.save_path)
    print(f"Saved: {args.save_path}")
    print("Keys:", list(out.keys()))
    if "pooled" in out:
        print(" pooled keys:", list(out["pooled"].keys()))


if __name__ == "__main__":
    main()
