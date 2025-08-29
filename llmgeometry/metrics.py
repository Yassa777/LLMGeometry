"""
Geometry-aware metrics with numerically stable fp32 compute.

Includes explained variance (EV), cross-entropy proxy (shift-invariant, bf16-safe),
causal EV via provided geometry, and logit-space EV given an unembedding matrix.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """EV = 1 - ||x - x_hat||^2 / ||x - mean(x)||^2 (computed in fp32)."""
    xf = x.detach().to(torch.float32)
    xhf = x_hat.detach().to(torch.float32)
    xm = xf.mean(dim=0, keepdim=True)
    num = torch.sum((xf - xhf) ** 2)
    den = torch.sum((xf - xm) ** 2) + 1e-8
    ev = 1.0 - (num / den)
    ev_val = float(ev.item())
    if not np.isfinite(ev_val):
        raise FloatingPointError("compute_explained_variance produced non-finite value")
    return ev_val


def compute_cross_entropy_proxy(x: torch.Tensor, x_hat: torch.Tensor, temperature: float = 1.0) -> float:
    """Cross-entropy proxy based on softmax over last dimension.

    Uses the identity CE(P,Q) = logsumexp(z) - sum(P * z), with z = x_hat / tau
    and P = softmax(x / tau). This is strictly shift-invariant and robust in bf16.
    """
    tau = float(temperature)
    xf = x.detach().to(torch.float32)
    xhf = x_hat.detach().to(torch.float32)
    p_true = F.softmax(xf / tau, dim=-1)
    z = xhf / tau
    lse = torch.logsumexp(z, dim=-1)
    dot = torch.sum(p_true * z, dim=-1)
    ce = (lse - dot).mean()
    val = float(ce.item())
    if not np.isfinite(val):
        raise FloatingPointError("compute_cross_entropy_proxy produced non-finite value")
    return val


def compute_dual_explained_variance(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    geometry=None,
    print_components: bool = False,
) -> Dict[str, float]:
    """Return standard EV and (optionally) causal EV under provided geometry."""
    xf = x.detach().to(torch.float32)
    xhf = x_hat.detach().to(torch.float32)
    xm = xf.mean(dim=0, keepdim=True)
    ev_standard = 1.0 - (torch.sum((xf - xhf) ** 2) / (torch.sum((xf - xm) ** 2) + 1e-8))
    results: Dict[str, float] = {"standard_ev": float(ev_standard.item())}

    if geometry is not None:
        xw = geometry.whiten(xf)
        xhw = geometry.whiten(xhf)
        xm_w = xw.mean(dim=0, keepdim=True)
        ev_causal = 1.0 - (torch.sum((xw - xhw) ** 2) / (torch.sum((xw - xm_w) ** 2) + 1e-8))
        results["causal_ev"] = float(ev_causal.item())

    return results


def compute_logit_ev(x: torch.Tensor, x_hat: torch.Tensor, unembedding: torch.Tensor) -> float:
    """EV in logit space: project via U, compute EV(X, X_hat)."""
    xf = x.detach().to(torch.float32)
    xhf = x_hat.detach().to(torch.float32)
    Uf = unembedding.detach().to(torch.float32)
    X = xf @ Uf.t()
    Xh = xhf @ Uf.t()
    Xm = X.mean(dim=0, keepdim=True)
    ev = 1.0 - (torch.sum((X - Xh) ** 2) / (torch.sum((X - Xm) ** 2) + 1e-8))
    val = float(ev.item())
    if not np.isfinite(val):
        raise FloatingPointError("compute_logit_ev produced non-finite value")
    return val

