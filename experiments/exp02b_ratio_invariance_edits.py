#!/usr/bin/env python3
"""
Exp02b: Ratio-invariance under actual parent edits with controls.

Computes sibling ratio KL(base||after) over child token sets under α∈{-1.0,-0.5,0,0.5,1.0}.
Controls:
 - Causal (normalize parent vector causally)
 - Euclidean (normalize with L2)
 - Random parent replacement (use another parent's vector)
 - Shuffled-U (permute unembedding rows to form W, normalize under that)

Writes per-parent and summary JSON with acceptance checks.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmgeometry import CausalGeometry
from llmgeometry.loaders import load_geometry, load_teacher_vectors
from llmgeometry.validation import sibling_ratio_kl_from_logits


def tokenize_ids(tok, tokens: List[str]) -> List[int]:
    ids: List[int] = []
    for t in tokens:
        out = tok(t, add_special_tokens=False).input_ids
        if out:
            ids.append(out[0])
    return sorted(list(set(ids)))


def permute_unembedding_W(geom: CausalGeometry, U: torch.Tensor) -> CausalGeometry:
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(U.shape[0], generator=rng)
    Up = U[perm]
    # Recompute W via ZCA on Up
    g = CausalGeometry(Up, shrinkage=0.0)
    return g


def main():
    ap = argparse.ArgumentParser(description="Exp02b: Ratio-invariance (edits)")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp02b"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    device = cfg["run"].get("device", "cpu")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    geom = load_geometry(cfg["inputs"]["geometry"])  # baseline causal geometry
    parents, child_vecs, child_deltas = load_teacher_vectors(cfg["inputs"]["teacher_vectors"])
    hier = json.load(open(cfg["inputs"]["hierarchies_json"]))

    # Prepare sibling token ids per parent
    sibling_ids: Dict[str, Dict[str, List[int]]] = {}
    parent_prompts: Dict[str, List[str]] = {}
    for h in hier:
        pid = h["parent"]["synset_id"]
        parent_prompts[pid] = h.get("parent_prompts", [])[: int(cfg.get("data", {}).get("prompts_per_parent", 8))]
        sib = {}
        for cid, toks in h.get("sibling_tokens", {}).items():
            sib[cid] = tokenize_ids(tok, toks)[:5]
        sibling_ids[pid] = sib

    # Conditions
    conds = ["causal", "euclid", "random_parent", "shuffled_U"]
    alphas = cfg.get("eval", {}).get("alphas", [-1.0, -0.5, 0.0, 0.5, 1.0])
    results: Dict[str, Dict] = {}

    # For shuffled-U control, get unembedding
    try:
        U = mdl.get_output_embeddings().weight.detach().to("cpu", dtype=torch.float32)
    except Exception:
        U = None
    geom_shuffled = permute_unembedding_W(geom, U) if U is not None else None

    pids = list(parents.keys())
    for pid in pids:
        if pid not in sibling_ids:
            continue
        prompts = parent_prompts.get(pid, [])
        if not prompts:
            continue
        pvec = parents[pid]
        # build random parent replacement if available
        rp_pid = random.choice([q for q in pids if q != pid]) if len(pids) > 1 else pid
        rp_vec = parents[rp_pid]

        per_cond = {}
        for cond in conds:
            per_alpha = {}
            for a in alphas:
                with torch.no_grad():
                    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                    out0 = mdl(**inputs)
                    logits0 = out0.logits.detach().cpu()
                # choose vector and normalization
                if cond == "causal":
                    v = geom.normalize_causal(torch.tensor(pvec))
                elif cond == "euclid":
                    v = torch.tensor(pvec, dtype=torch.float32)
                    v = v / (v.norm() + 1e-8)
                elif cond == "random_parent":
                    v = geom.normalize_causal(torch.tensor(rp_vec))
                elif cond == "shuffled_U" and geom_shuffled is not None:
                    v = geom_shuffled.normalize_causal(torch.tensor(pvec))
                else:
                    v = geom.normalize_causal(torch.tensor(pvec))

                # Apply edit at last token residual
                with torch.no_grad():
                    outputs = mdl(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1].clone()
                    last_idx = inputs["attention_mask"].sum(dim=1) - 1
                    vv = v.to(hidden.dtype).to(device)
                    hidden[torch.arange(hidden.size(0)), last_idx] += float(a) * vv
                    head = mdl.lm_head if hasattr(mdl, "lm_head") else mdl.get_output_embeddings()
                    logits1 = head(hidden).detach().cpu()

                # Aggregate sibling KLs
                sib = sibling_ids[pid]
                kls = []
                for cid, ids in sib.items():
                    if not ids:
                        continue
                    kl = sibling_ratio_kl_from_logits(logits0, logits1, ids)
                    kls.append(kl)
                med_kl = float(np.median(kls)) if kls else float("nan")
                per_alpha[str(a)] = {"median_kl": med_kl}
            per_cond[cond] = per_alpha
        results[pid] = per_cond

    # Summaries + acceptance
    summary = {}
    for cond in conds:
        all_kls = []
        for pid in results:
            for a, d in results[pid][cond].items():
                if float(a) == 0.0:
                    continue
                k = d.get("median_kl")
                if k is not None and np.isfinite(k):
                    all_kls.append(k)
        summary[cond] = {
            "median_kl": float(np.median(all_kls)) if all_kls else float("nan"),
            "n": len(all_kls),
        }
    acceptance = {
        "causal_pass": (summary.get("causal", {}).get("median_kl", 1.0) < 0.10),
        "euclid_pass": (summary.get("euclid", {}).get("median_kl", 0.0) > 0.25),
    }

    with open(out_dir / "ratio_invariance_edits.json", "w") as f:
        json.dump({"per_parent": results, "summary": summary, "acceptance": acceptance}, f, indent=2)
    print("Saved:", out_dir / "ratio_invariance_edits.json")


if __name__ == "__main__":
    main()

