"""
Geometry utilities for final-layer analysis in language models.

Implements ZCA whitening (Sigma, W), causal inner products and angles, and
diagnostics including the whitening invariant (W @ Sigma @ W^T ≈ I) and a
linear identity test (x → whiten → unwhiten ≈ x).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class CausalGeometry:
    """Handles causal inner products and ZCA whitening.

    Parameters
    ----------
    data_or_U : torch.Tensor
        2D tensor of shape [N, d] whose row covariance defines Sigma.
        Typical choices:
          - Unembedding rows (V x d)
          - Collected residual activations (B x d)
    shrinkage : float, default 0.05
        Ridge term added to Sigma for numerical stability.
    cov_device : str, default "cpu"
        Device to compute covariance and eigendecomposition.
    """

    def __init__(
        self,
        data_or_U: torch.Tensor,
        *,
        shrinkage: float = 0.05,
        cov_device: str = "cpu",
    ) -> None:
        self.shrinkage = float(shrinkage)
        self.cov_device = cov_device

        X = data_or_U.detach().to(cov_device, dtype=torch.float32)
        if X.ndim != 2:
            raise ValueError("data_or_U must be a [N, d] 2D tensor")
        N, d = X.shape
        Xc = X - X.mean(dim=0, keepdim=True)

        # Full covariance with shrinkage
        # torch.cov expects [d, N] input if using as cov of rows; use formula
        Sigma = (Xc.t() @ Xc) / max(1, N - 1)
        Sigma = Sigma + self.shrinkage * torch.eye(d, device=self.cov_device)

        # Eigh for SPD
        evals, evecs = torch.linalg.eigh(Sigma)
        evals = torch.clamp(evals, min=1e-8)
        D_inv_sqrt = torch.diag(evals.rsqrt())
        W = evecs @ D_inv_sqrt @ evecs.t()

        self.Sigma = Sigma  # [d, d]
        self.W = W  # [d, d]

    # -----------------------------
    # Basic operations
    # -----------------------------
    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Apply whitening: x̃ = x @ W^T (right-multiply by W^T).

        Casts to float32 for numerical stability and returns in the input dtype.
        """
        if x.ndim not in (1, 2):
            raise ValueError("whiten expects a 1D or 2D tensor")
        W = self.W.to(x.device, dtype=torch.float32)
        xf = x.to(torch.float32)
        out = xf @ W.t()
        return out.to(x.dtype)

    def unwhiten(self, x_whitened: torch.Tensor) -> torch.Tensor:
        """Apply inverse whitening via solve: W^T x = x̃ ⇒ x = (W^T)^{-1} x̃."""
        if x_whitened.ndim not in (1, 2):
            raise ValueError("unwhiten expects a 1D or 2D tensor")
        W = self.W.to(x_whitened.device, dtype=torch.float32)
        xw = x_whitened.to(torch.float32)
        # Solve (W^T) x = xw for x
        x_orig = torch.linalg.solve(W.t(), xw.t()).t()
        return x_orig.to(x_whitened.dtype)

    def causal_inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """⟨x,y⟩_c = (W x) · (W y). Supports 1D/2D inputs, returns per-row dot."""
        if x.ndim != y.ndim or x.shape[-1] != y.shape[-1]:
            raise ValueError("inputs must have matching dimensions")
        xw = self.whiten(x).to(torch.float32)
        yw = self.whiten(y).to(torch.float32)
        return torch.sum(xw * yw, dim=-1)

    def causal_norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.causal_inner_product(x, x))

    def causal_angle(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return angle in radians under the causal inner product."""
        ip = self.causal_inner_product(x, y)
        nx = self.causal_norm(x)
        ny = self.causal_norm(y)
        denom = nx * ny
        # Avoid division by zero — this will produce NaN if either vector is zero
        cos_theta = torch.clamp(ip / (denom + 1e-12), -1.0, 1.0)
        return torch.arccos(cos_theta)

    def normalize_causal(self, x: torch.Tensor) -> torch.Tensor:
        n = self.causal_norm(x)
        return x / (n + 1e-8)

    def to(self, device: str):
        self.Sigma = self.Sigma.to(device)
        self.W = self.W.to(device)
        return self

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def whitening_invariant_stats(self) -> Dict[str, float]:
        """Return diagnostics for C = W Σ W^T ≈ I."""
        W = self.W.detach().to(torch.float32, copy=True).cpu()
        Sigma = self.Sigma.detach().to(torch.float32, copy=True).cpu()
        I = torch.eye(W.shape[0], dtype=torch.float32)
        C = W @ Sigma @ W.t()
        E = C - I
        diag = torch.diag(C)
        off = C - torch.diag(diag)
        return {
            "whiten_diag_mean": float(diag.mean().item()),
            "whiten_diag_std": float(diag.std().item()),
            "whiten_offdiag_max": float(off.abs().max().item()),
            "whiten_offdiag_rms": float(torch.sqrt((off**2).mean()).item()),
            "whiten_fro_error": float(torch.linalg.norm(E, ord="fro").item()),
        }

    def test_linear_identity(self, x: torch.Tensor, tolerance: float = 1e-4) -> Dict[str, float]:
        """x → whiten → unwhiten; report mse, max_error, and EV.

        Always computes in float32 and returns floats.
        """
        xf = x.detach().to(torch.float32)
        xw = self.whiten(xf)
        xr = self.unwhiten(xw).to(torch.float32)
        err = xf - xr
        mse = torch.mean(err ** 2).item()
        max_err = torch.max(err.abs()).item()
        # EV = 1 - ||x - xr||^2 / ||x - mean||^2
        xm = xf.mean(dim=0, keepdim=True)
        ev = 1.0 - (torch.sum((xf - xr) ** 2) / (torch.sum((xf - xm) ** 2) + 1e-8)).item()
        return {
            "identity_mse": mse,
            "identity_max_error": max_err,
            "identity_ev": ev,
            "identity_ok": max_err < tolerance,
            "tolerance": float(tolerance),
        }

