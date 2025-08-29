#!/usr/bin/env python3
"""
Exp05b: Estimator shoot-out — LDA vs class-mean vs L2-probe.

For each concept with pos/neg activations, estimate directions using
 - LDAEstimator
 - MeanDiffEstimator
 - L2ProbeEstimator (logistic with L2)
Compute AUROC and angles to teacher vectors if provided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from llmgeometry import CausalGeometry
from llmgeometry.loaders import load_geometry, load_activations, save_json
from llmgeometry.estimators import LDAEstimator, MeanDiffEstimator, L2ProbeEstimator


def score_auc(direction: torch.Tensor, X_pos: torch.Tensor, X_neg: torch.Tensor, geom: CausalGeometry) -> float:
    w = geom.normalize_causal(direction.to(torch.float32))
    w_w = geom.whiten(w)
    xp = geom.whiten(X_pos.to(torch.float32))
    xn = geom.whiten(X_neg.to(torch.float32))
    s_pos = (xp * w_w).sum(dim=-1).cpu().numpy()
    s_neg = (xn * w_w).sum(dim=-1).cpu().numpy()
    y = np.concatenate([np.ones_like(s_pos), np.zeros_like(s_neg)])
    s = np.concatenate([s_pos, s_neg])
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description="Exp05b: Estimator shoot-out")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp05b"))
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = load_geometry(cfg["inputs"]["geometry"])  # geometry.pt from Exp01
    acts = load_activations(cfg["inputs"]["activations"])  # HDF5
    teacher_vecs = None
    if "teacher_vectors" in cfg.get("inputs", {}):
        tv = json.load(open(cfg["inputs"]["teacher_vectors"]))
        teacher_vecs = tv.get("parent_vectors", {})

    lda = LDAEstimator(shrinkage=float(cfg.get("geometry", {}).get("lda_shrinkage", 0.1)))
    md = MeanDiffEstimator()
    l2 = L2ProbeEstimator(C=float(cfg.get("probe", {}).get("C", 1.0)), max_iter=int(cfg.get("probe", {}).get("max_iter", 500)))

    per: Dict[str, Dict[str, float]] = {}
    auc_lda = []
    auc_md = []
    auc_l2 = []
    for cid, d in acts.items():
        if "pos" not in d or "neg" not in d or len(d["pos"]) == 0 or len(d["neg"]) == 0:
            continue
        w_lda = lda.estimate_binary_direction(d["pos"], d["neg"], geom, normalize=True)
        w_md = md.estimate_binary_direction(d["pos"], d["neg"], geom, normalize=True)
        w_l2 = l2.estimate_binary_direction(d["pos"], d["neg"], geom, normalize=True)
        a1 = score_auc(w_lda, d["pos"], d["neg"], geom)
        a2 = score_auc(w_md, d["pos"], d["neg"], geom)
        a3 = score_auc(w_l2, d["pos"], d["neg"], geom)
        res = {"auroc_lda": a1, "auroc_mean_diff": a2, "auroc_l2probe": a3}
        if teacher_vecs and cid in teacher_vecs:
            t = torch.tensor(teacher_vecs[cid], dtype=torch.float32)
            res.update({
                "angle_to_teacher_lda": float(torch.rad2deg(geom.causal_angle(w_lda, t)).item()),
                "angle_to_teacher_mean_diff": float(torch.rad2deg(geom.causal_angle(w_md, t)).item()),
                "angle_to_teacher_l2probe": float(torch.rad2deg(geom.causal_angle(w_l2, t)).item()),
            })
        per[cid] = res
        if np.isfinite(a1): auc_lda.append(a1)
        if np.isfinite(a2): auc_md.append(a2)
        if np.isfinite(a3): auc_l2.append(a3)

    summary = {
        "median_auroc_lda": float(np.median(auc_lda)) if auc_lda else float("nan"),
        "median_auroc_mean_diff": float(np.median(auc_md)) if auc_md else float("nan"),
        "median_auroc_l2probe": float(np.median(auc_l2)) if auc_l2 else float("nan"),
    }
    save_json({"per_concept": per, "summary": summary}, str(out_dir / "estimators.json"))
    print("Saved:", out_dir / "estimators.json")


if __name__ == "__main__":
    main()
