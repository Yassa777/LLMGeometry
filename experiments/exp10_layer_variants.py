#!/usr/bin/env python3
"""
Exp10: Layer variants — causal geometry from hidden-state covariances over a small prompt set.

For a list of layer indices, compute covariance of last hidden state of that
layer over N prompts, create W via ZCA (with shrinkage), and compare causal
angle medians using existing teacher vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmgeometry import CausalGeometry
from llmgeometry.loaders import load_teacher_vectors, save_json


def sample_prompts_from_json(hier_json: str, k: int = 50) -> List[str]:
    data = json.load(open(hier_json))
    out: List[str] = []
    for h in data:
        out.extend(h.get("parent_prompts", [])[:2])
        for cid, ps in h.get("child_prompts", {}).items():
            out.extend(ps[:1])
        if len(out) >= k:
            break
    return out[:k]


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
    ap = argparse.ArgumentParser(description="Exp10: Layer variants")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp10"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    device = cfg["run"].get("device", "cpu")
    layers: List[int] = cfg.get("eval", {}).get("layers", [])
    n_prompts = int(cfg.get("data", {}).get("n_prompts", 50))
    hier_json = cfg["inputs"]["hierarchies_json"]

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    parents, _, deltas = load_teacher_vectors(cfg["inputs"]["teacher_vectors"])
    prompts = sample_prompts_from_json(hier_json, k=n_prompts)

    # Collect hidden states for specified layers and compute covariance
    results = {}
    for p in prompts:
        inputs = tok(p, return_tensors="pt", truncation=True, max_length=64).to(device)
        out = mdl(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple of [B,T,d]
        if not results:
            for li in layers:
                results[li] = []
        for li in layers:
            H = hs[li].detach().float().cpu()  # [B,T,d]
            # take last token
            last = H[:, -1, :]  # [B, d]
            results[li].append(last)

    summaries = {}
    for li in layers:
        X = torch.cat(results[li], dim=0)  # [N, d]
        geom = CausalGeometry(X, shrinkage=float(cfg.get("geometry", {}).get("shrinkage", 0.05)))
        med = angle_median(geom, parents, deltas)
        inv = geom.whitening_invariant_stats()
        summaries[int(li)] = {**inv, "angle_median_deg": med}

    save_json({"layers": summaries}, str(out_dir / "layer_variants.json"))
    print("Saved:", out_dir / "layer_variants.json")


if __name__ == "__main__":
    main()
