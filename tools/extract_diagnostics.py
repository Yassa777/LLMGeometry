#!/usr/bin/env python3
"""
Geometry-only diagnostics aggregator.

Reads geometry and teacher vectors (from Exp01) and optional outputs from
Exp02/03 to produce a compact diagnostics JSON with whitening invariants,
angle stats, and ratio-invariance summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch


def load_geometry(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    W = data.get("W")
    Sigma = data.get("Sigma")
    if isinstance(W, torch.Tensor):
        W = W.float()
    if isinstance(Sigma, torch.Tensor):
        Sigma = Sigma.float()
    # Whitening invariants
    I = torch.eye(W.shape[0])
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


def main():
    ap = argparse.ArgumentParser(description="Extract geometry-only diagnostics")
    ap.add_argument("--base", type=str, default="runs/exp01")
    ap.add_argument("--out", type=str, default="runs/diagnostics.json")
    args = ap.parse_args()

    base = Path(args.base)
    geom_file = base / "geometry.pt"
    tv_file = base / "teacher_vectors.json"

    out: Dict[str, Any] = {}
    try:
        out.update(load_geometry(str(geom_file)))
    except Exception as e:
        out["geometry_error"] = str(e)

    # Angle stats from teacher vectors (Exp01 saved)
    try:
        tv = json.load(open(tv_file))
        out["angles_median_deg"] = tv.get("angle_stats", {}).get("median_angle_deg")
        out["angles_fraction_above_80"] = tv.get("angle_stats", {}).get("fraction_above_threshold")
    except Exception as e:
        out["angles_error"] = str(e)

    # Optional Exp02/Exp03 summaries
    exp02 = base.parent / "exp02" / "ratio_invariance.json"
    if exp02.exists():
        try:
            ri = json.load(open(exp02))
            out["ratio_invariance_median_kl"] = ri.get("summary", {}).get("median_kl")
            out["ratio_invariance_fraction_below_0_1"] = ri.get("summary", {}).get("fraction_below_0_1")
        except Exception as e:
            out["ratio_invariance_error"] = str(e)

    exp03 = base.parent / "exp03" / "euclid_vs_causal.json"
    if exp03.exists():
        try:
            ec = json.load(open(exp03))
            out["euclid_median_angle"] = ec.get("euclidean", {}).get("median")
            out["causal_median_angle"] = ec.get("causal", {}).get("median")
            out["causal_vs_euclid_improvement"] = ec.get("improvement", {}).get("median_improvement")
        except Exception as e:
            out["euclid_causal_error"] = str(e)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved diagnostics:", args.out)


if __name__ == "__main__":
    main()

