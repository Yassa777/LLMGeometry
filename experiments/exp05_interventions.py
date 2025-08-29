#!/usr/bin/env python3
"""
Exp05: Interventions suite (parent-vector edits) with a smoke metric.

For a small set of parents and prompts, apply parent-vector interventions across
several magnitudes and record mean |Δlogits| as a coarse effect-size proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmgeometry import steer_parent_vector
from llmgeometry.loaders import load_teacher_vectors, load_hierarchies


def main():
    ap = argparse.ArgumentParser(description="Exp05: Interventions")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp05"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    device = cfg["run"].get("device", "cpu")
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": device})
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load teacher vectors
    parents, _, _ = load_teacher_vectors(cfg["inputs"]["teacher_vectors"])
    hierarchies = load_hierarchies(cfg["inputs"]["hierarchies"])  # for prompts

    # Build a small prompt set per parent (first few parent prompts)
    parent_prompts = {}
    for h in hierarchies:
        pid = h.parent.synset_id
        if pid in parents:
            parent_prompts[pid] = h.parent_prompts[:3]

    magnitudes = cfg.get("eval", {}).get("magnitudes", [0.5, 1.0, 2.0])
    results = {}
    for pid, pvec in list(parents.items())[: min(5, len(parents))]:
        if pid not in parent_prompts:
            continue
        prompts = parent_prompts[pid]
        per_mag = []
        for m in magnitudes:
            out = steer_parent_vector(mdl, tok, prompts, pvec, magnitude=float(m), device=device)
            mad = torch.mean(torch.abs(out["logit_deltas"]))
            per_mag.append({"magnitude": float(m), "mean_abs_delta": float(mad.item())})
        results[pid] = per_mag

    with open(out_dir / "interventions.json", "w") as f:
        json.dump({"per_parent": results}, f, indent=2)
    print("Saved:", out_dir / "interventions.json")


if __name__ == "__main__":
    main()

