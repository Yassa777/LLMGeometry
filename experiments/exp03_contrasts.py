#!/usr/bin/env python3
"""
Exp03b: Contrasts ≈ mean-diff — LDA vs class-mean angle and AUROC.

Loads geometry, activations HDF5, computes per-concept directions via:
 - LDA (whitened)
 - Mean-difference (whitened)
Then compares angles between the two and evaluates AUROC per method.
Saves per-concept and summary JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from llmgeometry import CausalGeometry
from llmgeometry.loaders import load_geometry, load_activations, save_json
from llmgeometry.estimators import LDAEstimator, MeanDiffEstimator


def score_auc(direction: torch.Tensor, X_pos: torch.Tensor, X_neg: torch.Tensor, geom: CausalGeometry) -> float:
    # Score by causal inner product with direction (whitened dot)
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
    ap = argparse.ArgumentParser(description="Exp03b: LDA vs class-mean contrasts")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp03b"))
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = load_geometry(cfg["inputs"]["geometry"])  # geometry.pt from Exp01
    acts = load_activations(cfg["inputs"]["activations"])  # HDF5

    lda = LDAEstimator(shrinkage=float(cfg.get("geometry", {}).get("lda_shrinkage", 0.1)))
    md = MeanDiffEstimator()

    per: Dict[str, Dict[str, float]] = {}
    angles = []
    auc_lda = []
    auc_md = []
    for cid, d in acts.items():
        if "pos" not in d or "neg" not in d or len(d["pos"]) == 0 or len(d["neg"]) == 0:
            continue
        w_lda = lda.estimate_binary_direction(d["pos"], d["neg"], geom, normalize=True)
        w_md = md.estimate_binary_direction(d["pos"], d["neg"], geom, normalize=True)
        ang = torch.rad2deg(geom.causal_angle(w_lda, w_md)).item()
        auc1 = score_auc(w_lda, d["pos"], d["neg"], geom)
        auc2 = score_auc(w_md, d["pos"], d["neg"], geom)
        per[cid] = {"angle_deg": float(ang), "auroc_lda": auc1, "auroc_mean_diff": auc2}
        if np.isfinite(ang):
            angles.append(ang)
        if np.isfinite(auc1):
            auc_lda.append(auc1)
        if np.isfinite(auc2):
            auc_md.append(auc2)

    summary = {
        "median_angle_deg": float(np.median(angles)) if angles else float("nan"),
        "median_auroc_lda": float(np.median(auc_lda)) if auc_lda else float("nan"),
        "median_auroc_mean_diff": float(np.median(auc_md)) if auc_md else float("nan"),
    }
    save_json({"per_concept": per, "summary": summary}, str(out_dir / "contrasts.json"))
    print("Saved:", out_dir / "contrasts.json")


if __name__ == "__main__":
    main()
