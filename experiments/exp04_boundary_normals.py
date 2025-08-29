#!/usr/bin/env python3
"""
Exp04: Boundary-normal alignment.

For each parent p:
  - Build a child subspace from teacher child deltas (SVD, like Exp01)
  - Project activations into subspace
  - Fit binary logistic regressions one-vs-rest for each child c∈siblings(p)
  - Take the learned normal vector in subspace, map up via up-projector
  - Compare causal angle to teacher delta δ_{c|p}
Save per-parent median angles and an overall summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from scipy.optimize import nnls

from llmgeometry import CausalGeometry
from llmgeometry.loaders import load_geometry, load_teacher_vectors, load_activations, save_json


def fit_normals_for_parent(
    geom: CausalGeometry,
    parent_id: str,
    child_vecs: Dict[str, torch.Tensor],
    child_deltas: Dict[str, torch.Tensor],
    acts: Dict[str, Dict[str, torch.Tensor]],
    subspace_dim: int = 32,
) -> Dict[str, float]:
    """Return angles between logistic normals and teacher δ for each child in a parent."""
    # Build subspace from deltas
    if not child_deltas:
        return {}
    # Build subspace in whitened space to match projection space
    M = torch.stack([geom.whiten(v.to(torch.float32)) for v in child_deltas.values()])  # [n_children, d]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    V = Vh.t()
    k_energy = int((torch.cumsum(S, dim=0) / (S.sum() + 1e-12) <= 0.85).sum().item()) + 1
    k = min(subspace_dim, V.shape[1], k_energy)
    Vk_w = V[:, :k]           # [d, k] in whitened coords
    up = Vk_w                 # [d, k]

    # Prepare dataset: gather pos/neg per child c under this parent
    angles: Dict[str, float] = {}
    nnls_r2: Dict[str, float] = {}
    bary_r2: Dict[str, float] = {}
    for cid, cvec in child_vecs.items():
        # Skip if missing activations
        if cid not in acts:
            continue
        pos = acts[cid].get("pos")
        neg = acts[cid].get("neg")
        if pos is None or neg is None or len(pos) == 0 or len(neg) == 0:
            continue
        X = torch.cat([pos, neg], dim=0).to(torch.float32)
        y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).numpy()
        # Whiten and project to subspace
        Xw = geom.whiten(X)
        Xs = Xw @ up    # [N, k] whitened subspace

        # Fit logistic
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(Xs.numpy(), y)
        w_s = torch.tensor(clf.coef_[0], dtype=torch.float32)  # [k]
        # Normal in original space (causal): map up then normalize causally
        normal_w = (up @ w_s).to(torch.float32)
        normal_full = geom.unwhiten(normal_w)
        normal_full = geom.normalize_causal(normal_full)

        # Compare to teacher delta
        delta = geom.normalize_causal(child_deltas[cid].to(torch.float32))
        ang = torch.rad2deg(geom.causal_angle(normal_full, delta)).item()
        angles[cid] = ang

    # After gathering all children with activations, compute NNLS/barycentric R^2 on deltas
    valid_children = [cid for cid in child_deltas.keys() if cid in angles]
    if len(valid_children) >= 2:
        D = torch.stack([geom.normalize_causal(child_deltas[c].to(torch.float32)) for c in valid_children])  # [k, d]
        Dw = geom.whiten(D)  # work in causal space
        for i, cid in enumerate(valid_children):
            y = Dw[i].numpy()
            X = Dw[[j for j in range(len(valid_children)) if j != i]].numpy().T  # [d, k-1]
            coef, _ = nnls(X, y)
            y_hat = X @ coef
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-8)
            nnls_r2[cid] = 1.0 - ss_res / ss_tot
            # Barycentric: normalize NNLS coef to sum=1 if sum>0
            if coef.sum() > 1e-8:
                bcoef = coef / coef.sum()
                yb = X @ bcoef
                ss_res_b = float(np.sum((y - yb) ** 2))
                bary_r2[cid] = 1.0 - ss_res_b / ss_tot
            else:
                bary_r2[cid] = float("nan")

    return angles, nnls_r2, bary_r2


def main():
    ap = argparse.ArgumentParser(description="Exp04: Boundary-normal alignment")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp04"))
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = load_geometry(cfg["inputs"]["geometry"])  # geometry.pt from Exp01
    parents, child_vecs_all, child_deltas_all = load_teacher_vectors(cfg["inputs"]["teacher_vectors"])
    # Load hierarchical activations
    acts = load_activations(cfg["inputs"]["activations"])  # HDF5

    subspace_dim = int(cfg.get("geometry", {}).get("subspace_dim", 32))

    per_parent: Dict[str, Dict[str, float]] = {}
    medians: Dict[str, float] = {}
    for pid, child_vecs in child_vecs_all.items():
        angles, nnls_r2, bary_r2 = fit_normals_for_parent(
            geom,
            pid,
            child_vecs,
            child_deltas_all.get(pid, {}),
            acts,
            subspace_dim=subspace_dim,
        )
        if angles:
            per_parent[pid] = {"angles": angles, "nnls_r2": nnls_r2, "barycentric_r2": bary_r2}
            medians[pid] = float(np.median(list(angles.values())))

    summary = {
        "n_parents": len(per_parent),
        "median_of_medians": (float(np.median(list(medians.values()))) if medians else float("nan")),
        "per_parent_median": medians,
    }
    # Simplex/polytope summaries
    nnls_all = [v for d in per_parent.values() for v in d.get("nnls_r2", {}).values()]
    bary_all = [v for d in per_parent.values() for v in d.get("barycentric_r2", {}).values()]
    summary["nnls_r2_median"] = float(np.median(nnls_all)) if nnls_all else float("nan")
    summary["barycentric_r2_median"] = float(np.median(bary_all)) if bary_all else float("nan")

    out = {"per_parent": per_parent, "summary": summary}
    save_json(out, str(out_dir / "boundary_normals.json"))
    print("Saved:", out_dir / "boundary_normals.json")


if __name__ == "__main__":
    main()
