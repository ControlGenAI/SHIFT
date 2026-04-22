"""
Subspace-AcT building blocks:
- ContrastivePCA (mean-pool tokens per sample, then contrastive PCA on (N, C))
- ContrastivePCATokenWise (all token rows (N*T, C), same mu_neg / U API for (T, C) at inference)
- SinkhornTransport
- AffineTransport
"""

import math
from typing import Callable, Optional, Tuple

import torch


class ContrastivePCA:
    """
    Contrastive PCA for paired activations (neg/pos).

    Expected fit() input:
      neg_acts: (N, T, C)
      pos_acts: (N, T, C)

    pca_delta_mode (how to form rows of D before SVD on (N, C)):
      - "centered" (default): D = delta - mean(delta); mean paired shift is orthogonal to centered span.
      - "raw_delta": D = delta; first PCs can align with mean shift μ_δ.
      - "append_mean_dir": PCA on centered D, then extend U with μ_δ orthogonalized to U (final fit only).
    """

    def __init__(
        self,
        subspace_dim: Optional[int] = None,
        eps: float = 1e-8,
        pca_delta_mode: str = "centered",
    ):
        self.subspace_dim = subspace_dim
        self.eps = eps
        if pca_delta_mode not in ("centered", "raw_delta", "append_mean_dir"):
            raise ValueError(
                f"pca_delta_mode must be centered|raw_delta|append_mean_dir, got {pca_delta_mode!r}"
            )
        self.pca_delta_mode = pca_delta_mode
        self.U: Optional[torch.Tensor] = None          # (C, d)
        self.mu_neg: Optional[torch.Tensor] = None     # (C,)
        self._evals_desc: Optional[torch.Tensor] = None
        self._fitted = False

    def fit(
        self,
        neg_acts: torch.Tensor,
        pos_acts: torch.Tensor,
        *,
        finalize_append_mean: bool = False,
    ) -> "ContrastivePCA":
        if neg_acts.shape != pos_acts.shape:
            raise ValueError(
                f"neg_acts and pos_acts must have same shape, got "
                f"{tuple(neg_acts.shape)} vs {tuple(pos_acts.shape)}"
            )
        if neg_acts.dim() != 3:
            raise ValueError(
                f"Expected (N, T, C) tensors, got shape={tuple(neg_acts.shape)}"
            )

        neg = neg_acts.float()
        pos = pos_acts.float()

        # print(neg.shape, pos.shape)
        # print(neg.min(), neg.max(), neg.mean(), neg.std())
        # print(pos.min(), pos.max(), pos.mean(), pos.std())

        # Mean pool over tokens -> (N, C)
        neg_pool = neg.mean(dim=1)
        pos_pool = pos.mean(dim=1)

        #print(neg_pool.shape, pos_pool.shape)

        # Anchor in original space
        self.mu_neg = neg_pool.mean(dim=0)

        # Paired contrastive differences
        delta = pos_pool - neg_pool # (N, C)
        if self.pca_delta_mode == "raw_delta":
            D = delta
        else:
            # centered PCA (also the PCA step for append_mean_dir)
            D = delta - delta.mean(dim=0, keepdim=True)  # (N, C)
        n = D.shape[0]

        # Stable PCA: SVD on D (N x C), avoids MKL eigh crashes on C x C covariance.
        # Sigma = (D^T D) / N  => eigenvalues = S^2 / N, eigenvectors = V.
        _, s, v_t = torch.linalg.svd(D, full_matrices=False)
        evals = (s ** 2) / max(n, 1)
        evecs = v_t.T

        d_max = evecs.shape[1]
        d = d_max if self.subspace_dim is None else max(1, min(int(self.subspace_dim), d_max))
        U = evecs[:, :d]

        appended = False
        if (
            self.pca_delta_mode == "append_mean_dir"
            and finalize_append_mean
            and delta.shape[0] > 0
        ):
            mu_delta = delta.mean(dim=0)
            mu_norm = mu_delta.norm().clamp(min=self.eps)
            mu_dir = mu_delta / mu_norm
            mu_dir_orth = mu_dir - U @ (U.T @ mu_dir)
            o_norm = mu_dir_orth.norm()
            if o_norm > self.eps:
                mu_dir_orth = mu_dir_orth / o_norm.clamp(min=self.eps)
                U = torch.cat([U, mu_dir_orth.unsqueeze(1)], dim=1)
                last_e = evals[min(d - 1, evals.numel() - 1)]
                extra = (last_e * 1e-3).clamp(min=self.eps).unsqueeze(0)
                self._evals_desc = torch.cat([evals[:d], extra])
                appended = True

        if not appended:
            self._evals_desc = evals

        self.U = U
        self._fitted = True
        return self

    def explained_variance_ratio(self) -> torch.Tensor:
        if not self._fitted or self._evals_desc is None:
            raise RuntimeError("ContrastivePCA is not fitted. Call fit() first.")
        denom = self._evals_desc.sum().clamp(min=self.eps)
        return self._evals_desc / denom

    def project(self, acts: torch.Tensor) -> torch.Tensor:
        if not self._fitted or self.U is None or self.mu_neg is None:
            raise RuntimeError("ContrastivePCA is not fitted. Call fit() first.")
        return (acts.float() - self.mu_neg) @ self.U

    def unproject(self, z: torch.Tensor, delta_only: bool = True) -> torch.Tensor:
        if not self._fitted or self.U is None or self.mu_neg is None:
            raise RuntimeError("ContrastivePCA is not fitted. Call fit() first.")
        recon = z.float() @ self.U.T
        if delta_only:
            return recon
        return recon + self.mu_neg

    def get_projectors(
        self,
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        """
        Return lazy projectors to avoid materializing large CxC matrices.

        Returns:
          (P, P_perp) as callables over acts[..., C]
        """
        if not self._fitted or self.U is None or self.mu_neg is None:
            raise RuntimeError("ContrastivePCA is not fitted. Call fit() first.")

        def proj_in(acts: torch.Tensor) -> torch.Tensor:
            z = self.project(acts)
            return self.unproject(z, delta_only=True)

        def proj_perp(acts: torch.Tensor) -> torch.Tensor:
            centered = acts.float() - self.mu_neg
            return centered - proj_in(acts)

        return proj_in, proj_perp


class ContrastivePCATokenWise:
    """
    Contrastive PCA without mean-pooling over the token axis.

    fit() expects:
      neg_acts, pos_acts: (N, T, C)

    Rows fed to PCA are per-token mean paired differences:
      delta[t, :] = mean_n(pos[n, t, :] - neg[n, t, :])  ->  (T, C)

    Anchor ``mu_neg`` is tokenwise mean over batch:
      mu_neg[t, :] = mean_n(neg[n, t, :])  ->  (T, C)

    At inference, project full activations A ∈ R^{T×C} with:
      Z = (A - mu_neg) @ U
    (and the same for batched (B, T, C)).
    """

    def __init__(
        self,
        subspace_dim: Optional[int] = None,
        eps: float = 1e-8,
        pca_delta_mode: str = "centered",
    ):
        self.subspace_dim = subspace_dim
        self.eps = eps
        if pca_delta_mode not in ("centered", "raw_delta", "append_mean_dir"):
            raise ValueError(
                f"pca_delta_mode must be centered|raw_delta|append_mean_dir, got {pca_delta_mode!r}"
            )
        self.pca_delta_mode = pca_delta_mode
        self.U: Optional[torch.Tensor] = None
        self.mu_neg: Optional[torch.Tensor] = None
        self._evals_desc: Optional[torch.Tensor] = None
        self._fitted = False

    def fit(
        self,
        neg_acts: torch.Tensor,
        pos_acts: torch.Tensor,
        *,
        finalize_append_mean: bool = False,
    ) -> "ContrastivePCATokenWise":
        if neg_acts.shape != pos_acts.shape:
            raise ValueError(
                f"neg_acts and pos_acts must have same shape, got "
                f"{tuple(neg_acts.shape)} vs {tuple(pos_acts.shape)}"
            )
        if neg_acts.dim() != 3:
            raise ValueError(
                f"Expected (N, T, C) tensors, got shape={tuple(neg_acts.shape)}"
            )

        neg = neg_acts.float()
        pos = pos_acts.float()

        self.mu_neg = neg.mean(dim=(0))

        delta = (pos - neg).mean(dim=0)
        if self.pca_delta_mode == "raw_delta":
            D = delta.reshape(-1, delta.shape[-1])
        else:
            flat = delta.reshape(-1, delta.shape[-1])
            D = flat - flat.mean(dim=0, keepdim=True)

        n = D.shape[0]
        _, s, v_t = torch.linalg.svd(D, full_matrices=False)
        evals = (s ** 2) / max(n, 1)
        evecs = v_t.T

        d_max = evecs.shape[1]
        d = d_max if self.subspace_dim is None else max(1, min(int(self.subspace_dim), d_max))
        U = evecs[:, :d]

        appended = False
        if (
            self.pca_delta_mode == "append_mean_dir"
            and finalize_append_mean
            and delta.numel() > 0
        ):
            mu_delta = delta.mean(dim=(0))
            mu_norm = mu_delta.norm().clamp(min=self.eps)
            mu_dir = mu_delta / mu_norm
            mu_dir_orth = mu_dir - U @ (U.T @ mu_dir)
            o_norm = mu_dir_orth.norm()
            if o_norm > self.eps:
                mu_dir_orth = mu_dir_orth / o_norm.clamp(min=self.eps)
                U = torch.cat([U, mu_dir_orth.unsqueeze(1)], dim=1)
                last_e = evals[min(d - 1, evals.numel() - 1)]
                extra = (last_e * 1e-3).clamp(min=self.eps).unsqueeze(0)
                self._evals_desc = torch.cat([evals[:d], extra])
                appended = True

        if not appended:
            self._evals_desc = evals

        self.U = U
        self._fitted = True
        return self

    def explained_variance_ratio(self) -> torch.Tensor:
        if not self._fitted or self._evals_desc is None:
            raise RuntimeError("ContrastivePCATokenWise is not fitted. Call fit() first.")
        denom = self._evals_desc.sum().clamp(min=self.eps)
        return self._evals_desc / denom

    def project(self, acts: torch.Tensor) -> torch.Tensor:
        if not self._fitted or self.U is None or self.mu_neg is None:
            raise RuntimeError("ContrastivePCATokenWise is not fitted. Call fit() first.")
        return (acts.float() - self.mu_neg) @ self.U

    def unproject(self, z: torch.Tensor, delta_only: bool = True) -> torch.Tensor:
        if not self._fitted or self.U is None or self.mu_neg is None:
            raise RuntimeError("ContrastivePCATokenWise is not fitted. Call fit() first.")
        recon = z.float() @ self.U.T
        if delta_only:
            return recon
        return recon + self.mu_neg

    def get_projectors(
        self,
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        if not self._fitted or self.U is None or self.mu_neg is None:
            raise RuntimeError("ContrastivePCATokenWise is not fitted. Call fit() first.")

        def proj_in(acts: torch.Tensor) -> torch.Tensor:
            z = self.project(acts)
            return self.unproject(z, delta_only=True)

        def proj_perp(acts: torch.Tensor) -> torch.Tensor:
            centered = acts.float() - self.mu_neg
            return centered - proj_in(acts)

        return proj_in, proj_perp


class SinkhornTransport:
    """
    Entropic OT transport in subspace coordinates.

    Out-of-sample extension uses the dual potential ``g`` of the target measure:

        T(z) = softmax_j( g[j] - ||z - y_j||^2 / eps ) @ Y

    so we do NOT store the full coupling matrix (memory O(N*M)) — only ``Y``,
    ``g`` and ``eps`` (memory O(m*d + m)).  This is the standard continuous
    barycentric extension and gives a smooth map (vs. piecewise-constant NN
    barycenter that the previous version produced).

    Iterations are performed in float64 by default for numerical stability;
    fitted tensors are stored in float32.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_iter: int = 100,
        tol: float = 1e-6,
        inner_dtype: torch.dtype = torch.float64,
    ):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.inner_dtype = inner_dtype

        self.z_pos: Optional[torch.Tensor] = None  # (m, d), float32
        self.g: Optional[torch.Tensor] = None      # (m,), float32  target dual potential
        # Kept for inspection/backward-compat; not used by transport().
        self.z_neg: Optional[torch.Tensor] = None
        self.f: Optional[torch.Tensor] = None

    def fit(self, z_neg: torch.Tensor, z_pos: torch.Tensor) -> "SinkhornTransport":
        if z_neg.dim() != 2 or z_pos.dim() != 2:
            raise ValueError("SinkhornTransport.fit expects 2D tensors (N, d).")
        if z_neg.shape[1] != z_pos.shape[1]:
            raise ValueError(f"Mismatched feature dims: {z_neg.shape} vs {z_pos.shape}")
        n, m = z_neg.shape[0], z_pos.shape[0]
        if n < 1 or m < 1:
            raise ValueError(f"Empty measures: n={n}, m={m}")

        # All Sinkhorn math in inner_dtype (default fp64) for stability.
        X = z_neg.to(self.inner_dtype)
        Y = z_pos.to(self.inner_dtype)
        device = X.device

        C = torch.cdist(X, Y, p=2).pow(2)
        log_K = -C / self.epsilon
        log_mu = torch.full((n,), -math.log(float(n)),
                            device=device, dtype=self.inner_dtype)
        log_nu = torch.full((m,), -math.log(float(m)),
                            device=device, dtype=self.inner_dtype)

        f = torch.zeros(n, device=device, dtype=self.inner_dtype)
        g = torch.zeros(m, device=device, dtype=self.inner_dtype)
        for _ in range(self.max_iter):
            f_prev = f
            g_prev = g
            f = log_mu - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
            g = log_nu - torch.logsumexp(log_K + f.unsqueeze(1), dim=0)
            delta = max((f - f_prev).abs().max().item(),
                        (g - g_prev).abs().max().item())
            if delta < self.tol:
                break

        self.z_pos = Y.to(torch.float32)
        self.z_neg = X.to(torch.float32)
        self.g = g.to(torch.float32)
        self.f = f.to(torch.float32)
        return self

    def transport(self, z: torch.Tensor) -> torch.Tensor:
        if self.z_pos is None or self.g is None:
            raise RuntimeError("SinkhornTransport is not fitted. Call fit() first.")
        zq = z.float()
        single = zq.dim() == 1
        if single:
            zq = zq.unsqueeze(0)
        Y = self.z_pos.to(zq.device)
        g = self.g.to(zq.device)
        sq_dist = torch.cdist(zq, Y, p=2).pow(2)            # (Q, M)
        log_w = g.unsqueeze(0) - sq_dist / self.epsilon      # (Q, M)
        weights = torch.softmax(log_w, dim=1)                # (Q, M)
        out = weights @ Y                                    # (Q, d)
        return out.squeeze(0) if single else out


class AffineTransport:
    """
    Closed-form affine transport ``T(z) = W z + b`` in subspace coordinates.

    Solves a ridge-regularised linear regression on paired samples
    ``(z_neg_i, z_pos_i)``.  This is NOT the Bures (Gaussian-OT) map — it is
    the L2-best linear regressor — and assumes the rows of ``z_neg`` and
    ``z_pos`` are paired in some meaningful way (e.g. same prompt-pair index).
    """

    def __init__(self, ridge: float = 1e-5, scale_ridge: bool = True):
        """
        Parameters
        ----------
        ridge : float
            Tikhonov coefficient.
        scale_ridge : bool
            If True (default) the effective ridge is
            ``ridge * mean(diag(Cov_xx))``, making the regularizer invariant
            to input scale.  Recommended.
        """
        if ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        self.ridge = float(ridge)
        self.scale_ridge = bool(scale_ridge)
        self.W: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None

    def fit(self, z_neg: torch.Tensor, z_pos: torch.Tensor) -> "AffineTransport":
        if z_neg.dim() != 2 or z_pos.dim() != 2:
            raise ValueError("AffineTransport.fit expects 2D tensors (N, d).")
        if z_neg.shape != z_pos.shape:
            raise ValueError(f"Shape mismatch: {z_neg.shape} vs {z_pos.shape}")

        X = z_neg.float()
        Y = z_pos.float()
        n, d = X.shape
        if n < 2:
            raise ValueError("Need at least 2 samples for affine transport fit.")

        x_mean = X.mean(dim=0)
        y_mean = Y.mean(dim=0)
        Xc = X - x_mean
        Yc = Y - y_mean

        C_pm = (Yc.T @ Xc) / n
        C_mm = (Xc.T @ Xc) / n
        eye = torch.eye(d, device=X.device, dtype=X.dtype)

        if self.scale_ridge:
            diag_mean = C_mm.diagonal().mean().clamp(min=1e-12)
            ridge_eff = self.ridge * diag_mean
        else:
            ridge_eff = torch.tensor(self.ridge, device=X.device, dtype=X.dtype)
        C_mm_reg = C_mm + ridge_eff * eye

        # W satisfies  C_mm_reg @ W.T = C_pm.T
        # Use solve (more stable than inv) on a symmetric PSD-ish system.
        W_t = torch.linalg.solve(C_mm_reg, C_pm.T)
        self.W = W_t.T.contiguous()
        self.b = y_mean - self.W @ x_mean
        return self

    def transport(self, z: torch.Tensor) -> torch.Tensor:
        if self.W is None or self.b is None:
            raise RuntimeError("AffineTransport is not fitted. Call fit() first.")
        zf = z.float()
        return zf @ self.W.T + self.b
