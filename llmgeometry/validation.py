"""
Validation helpers for geometry-only experiments.

Includes:
- Hierarchical orthogonality test (angles between parent vectors and child deltas)
- Synthetic ratio-invariance test (distribution over child magnitudes preserved
  under interventions along the parent vector in causal space)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _angles_parent_child(
    parent_vectors: Dict[str, torch.Tensor],
    child_deltas: Dict[str, Dict[str, torch.Tensor]],
    geometry,
) -> List[float]:
    """Compute causal angles (in degrees) between ℓ_p and δ_{c|p} across hierarchy."""
    angles_deg: List[float] = []
    for pid, deltas in child_deltas.items():
        if pid not in parent_vectors:
            continue
        p = parent_vectors[pid]
        for cid, d in deltas.items():
            if torch.norm(d) < 1e-8:
                continue
            ang = geometry.causal_angle(p, d)
            angles_deg.append(float(torch.rad2deg(ang).item()))
    return angles_deg


def hierarchical_orthogonality(
    parent_vectors: Dict[str, torch.Tensor],
    child_deltas: Dict[str, Dict[str, torch.Tensor]],
    geometry,
    *,
    threshold_deg: float = 80.0,
) -> Dict[str, float]:
    """Return angle statistics for ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0 across the hierarchy."""
    angles_deg = _angles_parent_child(parent_vectors, child_deltas, geometry)
    if not angles_deg:
        return {
            "n_pairs": 0,
            "median_angle_deg": float("nan"),
            "mean_angle_deg": float("nan"),
            "std_angle_deg": float("nan"),
            "q25_angle_deg": float("nan"),
            "q75_angle_deg": float("nan"),
            "fraction_above_threshold": 0.0,
            "fraction_above_85deg": 0.0,
        }
    arr = np.array(angles_deg)
    return {
        "n_pairs": int(arr.size),
        "median_angle_deg": float(np.median(arr)),
        "mean_angle_deg": float(np.mean(arr)),
        "std_angle_deg": float(np.std(arr)),
        "q25_angle_deg": float(np.percentile(arr, 25)),
        "q75_angle_deg": float(np.percentile(arr, 75)),
        "fraction_above_threshold": float(np.mean(arr >= threshold_deg)),
        "fraction_above_85deg": float(np.mean(arr >= 85.0)),
    }


def ratio_invariance_synthetic(
    parent_vector: torch.Tensor,
    child_vectors: List[torch.Tensor],
    geometry,
    *,
    alphas: List[float] = (0.5, 1.0, 2.0),
) -> Dict[str, object]:
    """Synthetic ratio-invariance using vector magnitudes in causal space.

    For a given parent ℓ_p and set of child vectors {ℓ_c}, compute baseline
    distribution over child magnitudes (softmax of ||W ℓ_c||), then apply
    interventions ℓ_c' = ℓ_c + α ℓ_p in causal space and compare distributions
    via KL (baseline || intervention). Lower KL indicates better invariance.
    """
    if not child_vectors:
        return {"by_alpha": {}, "aggregate": {"median_kl": float("nan"), "n_tests": 0}}

    # Work in causal (whitened) space
    pw = geometry.whiten(parent_vector.to(torch.float32))
    Cw = [geometry.whiten(c.to(torch.float32)) for c in child_vectors]

    base_proj = torch.stack([torch.norm(cw, p=2) for cw in Cw])
    base_probs = torch.softmax(base_proj, dim=0)

    by_alpha: Dict[float, Dict[str, float]] = {}
    kls: List[float] = []
    for a in alphas:
        int_proj = torch.stack([torch.norm(cw + a * pw, p=2) for cw in Cw])
        int_probs = torch.softmax(int_proj, dim=0)
        # KL(base || int) = sum base * (log base - log int)
        kl = torch.sum(base_probs * (torch.log(base_probs + 1e-8) - torch.log(int_probs + 1e-8))).item()
        by_alpha[float(a)] = {"kl_divergence": float(kl)}
        kls.append(kl)

    agg = {
        "median_kl": float(np.median(kls)) if kls else float("nan"),
        "fraction_below_0_1": float(np.mean(np.array(kls) < 0.1)) if kls else 0.0,
        "n_tests": len(kls),
    }
    return {"by_alpha": by_alpha, "aggregate": agg}
