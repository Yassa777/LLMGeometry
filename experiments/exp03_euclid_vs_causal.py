#!/usr/bin/env python3
"""
Exp03: Euclidean vs Causal angle distributions.

Loads teacher_vectors.json and geometry.pt, compares Euclidean and causal angle
statistics (median, fraction ≥ 80°) and saves a JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from llmgeometry import CausalGeometry


def main():
    ap = argparse.ArgumentParser(description="Exp03: Euclid vs Causal angles")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp03"))
    out_dir.mkdir(parents=True, exist_ok=True)

    gpath = Path(cfg["inputs"]["geometry"])  # geometry.pt from Exp01
    tpath = Path(cfg["inputs"]["teacher_vectors"])  # teacher_vectors.json from Exp01
    g = torch.load(gpath, map_location="cpu")
    geom = CausalGeometry(torch.eye(len(g["W"])) )
    geom.Sigma = g["Sigma"].float()
    geom.W = g["W"].float()

    data = json.load(open(tpath))
    parents = {k: torch.tensor(v, dtype=torch.float32) for k, v in data["parent_vectors"].items()}
    deltas = {pid: {cid: torch.tensor(v, dtype=torch.float32) for cid, v in d.items()} for pid, d in data["child_deltas"].items()}

    euclid_angles = []
    causal_angles = []
    for pid, child_map in deltas.items():
        if pid not in parents:
            continue
        p = parents[pid]
        for cid, d in child_map.items():
            # Euclid
            cos_sim = torch.nn.functional.cosine_similarity(p, d, dim=0)
            euclid = torch.rad2deg(torch.arccos(torch.clamp(cos_sim, -1, 1))).item()
            euclid_angles.append(euclid)
            # Causal
            ang = geom.causal_angle(p, d)
            causal_angles.append(float(torch.rad2deg(ang).item()))

    eu = np.array(euclid_angles)
    ca = np.array(causal_angles)
    out = {
        "euclidean": {
            "median": float(np.median(eu)) if eu.size else float("nan"),
            "fraction_above_80": float(np.mean(eu >= 80)) if eu.size else 0.0,
        },
        "causal": {
            "median": float(np.median(ca)) if ca.size else float("nan"),
            "fraction_above_80": float(np.mean(ca >= 80)) if ca.size else 0.0,
        },
        "improvement": {
            "median_improvement": (float(np.median(ca) - np.median(eu)) if eu.size and ca.size else float("nan")),
            "orthogonality_improvement": (float(np.mean(ca >= 80) - np.mean(eu >= 80)) if eu.size and ca.size else float("nan")),
        },
    }

    with open(out_dir / "euclid_vs_causal.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", out_dir / "euclid_vs_causal.json")


if __name__ == "__main__":
    main()

