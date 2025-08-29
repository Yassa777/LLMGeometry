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
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from llmgeometry import CausalGeometry
from llmgeometry.loaders import load_geometry, load_teacher_vectors, load_hierarchies, load_activations, save_json


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
    M = torch.stack(list(child_deltas.values()))  # [n_children, d]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    V = Vh.t()
    k_energy = int((torch.cumsum(S, dim=0) / (S.sum() + 1e-12) <= 0.85).sum().item()) + 1
    k = min(subspace_dim, V.shape[1], k_energy)
    Vk = V[:, :k]             # [d, k]
    down = Vk.t()             # [k, d]
    up = Vk                   # [d, k]

    # Prepare dataset: gather pos/neg per child c under this parent
    angles: Dict[str, float] = {}
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
        Xs = Xw @ up    # [N, k]

        # Fit logistic
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(Xs.numpy(), y)
        w_s = torch.tensor(clf.coef_[0], dtype=torch.float32)  # [k]
        # Normal in original space (causal): map up then normalize causally
        normal_full = (up @ w_s).to(torch.float32)
        normal_full = geom.normalize_causal(normal_full)

        # Compare to teacher delta
        delta = geom.normalize_causal(child_deltas[cid].to(torch.float32))
        ang = torch.rad2deg(geom.causal_angle(normal_full, delta)).item()
        angles[cid] = ang

    return angles


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
        angles = fit_normals_for_parent(
            geom,
            pid,
            child_vecs,
            child_deltas_all.get(pid, {}),
            acts,
            subspace_dim=subspace_dim,
        )
        if angles:
            per_parent[pid] = angles
            medians[pid] = float(np.median(list(angles.values())))

    summary = {
        "n_parents": len(per_parent),
        "median_of_medians": (float(np.median(list(medians.values()))) if medians else float("nan")),
        "per_parent_median": medians,
    }
    out = {"per_parent": per_parent, "summary": summary}
    save_json(out, str(out_dir / "boundary_normals.json"))
    print("Saved:", out_dir / "boundary_normals.json")


if __name__ == "__main__":
    main()

