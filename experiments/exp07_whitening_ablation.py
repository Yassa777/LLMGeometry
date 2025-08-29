#!/usr/bin/env python3
"""
Exp07: Whitening shrinkage ablation.

Recompute geometry from unembedding with different shrinkage values and compare
whitening invariants + causal angle medians using existing teacher vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM

from llmgeometry import CausalGeometry


def load_unembedding(model_name: str, device: str = "cpu") -> torch.Tensor:
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
    if hasattr(mdl, "lm_head") and isinstance(mdl.lm_head, torch.nn.Module):
        U = mdl.lm_head.weight.detach().to("cpu")
    else:
        U = mdl.get_output_embeddings().weight.detach().to("cpu")
    del mdl
    return U


def angle_median(geom: CausalGeometry, parents: dict, deltas: dict) -> float:
    angs: List[float] = []
    for pid, cd in deltas.items():
        if pid not in parents:
            continue
        p = torch.tensor(parents[pid], dtype=torch.float32)
        for cid, d in cd.items():
            delt = torch.tensor(d, dtype=torch.float32)
            angs.append(float(torch.rad2deg(geom.causal_angle(p, delt)).item()))
    return float(np.median(angs)) if angs else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Exp07: Whitening ablation")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp07"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    device = cfg["run"].get("device", "cpu")
    U = load_unembedding(model_name, device=device)

    teacher = json.load(open(cfg["inputs"]["teacher_vectors"]))
    parents = teacher["parent_vectors"]
    deltas = teacher["child_deltas"]

    shrinkages = cfg.get("geometry", {}).get("shrinkages", [0.0, 0.01, 0.05, 0.1])
    results = {}
    for s in shrinkages:
        geom = CausalGeometry(U, shrinkage=float(s))
        inv = geom.whitening_invariant_stats()
        med = angle_median(geom, parents, deltas)
        results[str(s)] = {**inv, "angle_median_deg": med}

    with open(out_dir / "whitening_ablation.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
    print("Saved:", out_dir / "whitening_ablation.json")


if __name__ == "__main__":
    main()
