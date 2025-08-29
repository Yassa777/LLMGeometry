#!/usr/bin/env python3
"""
Exp02: Ratio-invariance (synthetic magnitude-based check in causal space).

Loads teacher_vectors.json from Exp01 and computes per-parent ratio-invariance
KLs for magnitudes alphas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from llmgeometry import CausalGeometry, ratio_invariance_synthetic


def main():
    ap = argparse.ArgumentParser(description="Exp02: Ratio-invariance synthetic")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp02"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load geometry and teacher vectors
    gpath = Path(cfg["inputs"]["geometry"])  # path to geometry.pt from Exp01
    tpath = Path(cfg["inputs"]["teacher_vectors"])  # teacher_vectors.json from Exp01
    g = torch.load(gpath, map_location="cpu")
    geom = CausalGeometry(torch.eye(len(g["W"])) )
    geom.Sigma = g["Sigma"].float()
    geom.W = g["W"].float()

    data = json.load(open(tpath))
    parents = {k: torch.tensor(v, dtype=torch.float32) for k, v in data["parent_vectors"].items()}
    child_vecs = {pid: {cid: torch.tensor(v, dtype=torch.float32) for cid, v in d.items()} for pid, d in data["child_vectors"].items()}

    alphas = cfg.get("eval", {}).get("alphas", [0.5, 1.0, 2.0])
    results = {}
    all_kls = []
    for pid, pvec in parents.items():
        children = list(child_vecs.get(pid, {}).values())
        if not children:
            continue
        res = ratio_invariance_synthetic(pvec, children, geom, alphas=alphas)
        results[pid] = res
        for v in res["by_alpha"].values():
            all_kls.append(v["kl_divergence"])

    summary = {
        "median_kl": float(torch.median(torch.tensor(all_kls)).item()) if all_kls else float("nan"),
        "fraction_below_0_1": float((torch.tensor(all_kls) < 0.1).float().mean().item()) if all_kls else 0.0,
    }
    out = {"per_parent": results, "summary": summary}
    with open(out_dir / "ratio_invariance.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", out_dir / "ratio_invariance.json")


if __name__ == "__main__":
    main()

