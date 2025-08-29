#!/usr/bin/env python3
"""
Exp06: Fisher/logit diagonal approximations for geometry.

Compute a diagonal approximation to Fisher (E[p(1-p)]) or logit covariance
over a small prompt sample, form a diagonal W, and compare causal angle stats
to the baseline geometry.
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
from llmgeometry.loaders import load_teacher_vectors, load_geometry, save_json


def sample_prompts(hierarchies, k: int = 50) -> List[str]:
    """Collect a small prompt set from hierarchy JSON (dicts or dataclass-serialized)."""
    out: List[str] = []
    for h in hierarchies:
        parent_prompts = h.get("parent_prompts", []) if isinstance(h, dict) else getattr(h, "parent_prompts", [])
        out.extend(parent_prompts[:2])
        child_prompts = h.get("child_prompts", {}) if isinstance(h, dict) else getattr(h, "child_prompts", {})
        for ps in child_prompts.values():
            out.extend(ps[:1])
        if len(out) >= k:
            break
    return out[:k]


def main():
    ap = argparse.ArgumentParser(description="Exp06: Fisher/logit approximations")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp06"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    device = cfg["run"].get("device", "cpu")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    parents, child_vecs, child_deltas = load_teacher_vectors(cfg["inputs"]["teacher_vectors"])
    base_geom = load_geometry(cfg["inputs"]["geometry"])  # baseline geometry
    hier = json.load(open(cfg["inputs"]["hierarchies_json"]))

    # Build prompt set
    prompts = sample_prompts(hier, k=int(cfg.get("data", {}).get("n_prompts", 50)))
    # Collect last-token probs/logits
    probs = []
    logits = []
    with torch.no_grad():
        for p in prompts:
            inputs = tok(p, return_tensors="pt", truncation=True, max_length=64).to(device)
            out = mdl(**inputs)
            last = out.logits[:, -1, :].float().cpu()
            logits.append(last)
            probs.append(torch.softmax(last, dim=-1))
    P = torch.cat(probs, dim=0)      # [N, V]
    L = torch.cat(logits, dim=0)     # [N, V]

    # Diagonal Fisher & logit variance
    fisher_diag = (P * (1 - P)).mean(dim=0) + 1e-5
    var_diag = (L - L.mean(dim=0, keepdim=True)).pow(2).mean(dim=0) + 1e-5

    # Map vocab-diagonal stats to residual-diagonal via U (unembedding)
    # diag(U^T diag(d) U)[j] = sum_v d[v] * U[v,j]^2
    with torch.no_grad():
        U = mdl.get_output_embeddings().weight.detach().to("cpu", dtype=torch.float32)  # [V, d]
    Ud2 = (U ** 2).t()  # [d, V]
    d_f = Ud2 @ fisher_diag  # [d]
    d_l = Ud2 @ var_diag     # [d]
    d_f = d_f + 1e-5
    d_l = d_l + 1e-5

    # Form diagonal W approximations in residual space
    W_f = torch.diag(1.0 / torch.sqrt(d_f))
    W_l = torch.diag(1.0 / torch.sqrt(d_l))

    # Compare to baseline geometry via a rough angle check on a few deltas
    def angle_summary(Wdiag):
        # Construct a causal geometry with diagonal Sigma consistent with W
        d = Wdiag.shape[0]
        geom = CausalGeometry(torch.eye(d), shrinkage=0.0)
        geom.W = Wdiag.to(torch.float32)
        # If W = diag(inv_sqrt(var)), then Sigma ≈ diag(var)
        var = torch.diag(Wdiag).pow(-2)
        geom.Sigma = torch.diag(var)
        # angles between a few parent/deltas (fallback to causal norms of deltas)
        angs = []
        count = 0
        for pid, deltas in child_deltas.items():
            p = parents.get(pid)
            if p is None:
                continue
            for cid, delt in deltas.items():
                ang = torch.rad2deg(geom.causal_angle(p, delt)).item()
                angs.append(ang)
                count += 1
                if count >= 100:
                    break
            if count >= 100:
                break
        if not angs:
            return {"median": float("nan"), "fraction_above_80": 0.0}
        a = np.array(angs)
        return {"median": float(np.median(a)), "fraction_above_80": float(np.mean(a >= 80))}

    summary = {
        "baseline": angle_summary(base_geom.W.cpu()),
        "fisher_diag": angle_summary(W_f),
        "logit_var_diag": angle_summary(W_l),
    }
    save_json(summary, str(out_dir / "fisher_logit_summary.json"))
    print("Saved:", out_dir / "fisher_logit_summary.json")


if __name__ == "__main__":
    main()
