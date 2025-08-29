#!/usr/bin/env python3
"""
Exp10b: Emergence curves with unified causal metric.

Uses fixed causal geometry from Exp01 (unembedding-based W) for all layers.
For each layer L in eval.layers:
  - Angle vs layer: for each child concept, fit LDA on last-token activations
    at layer L (pos vs sibling-neg), compute angle to teacher delta (Exp01),
    aggregate median across children.
  - KL vs layer: for each parent, apply a small edit +αℓ_p at layer L via hook
    and compute sibling ratio KL(base||after) over child token sets, aggregate
    median across parents.

Outputs runs/exp10b/emergence_curves.json with curves and summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmgeometry import CausalGeometry
from llmgeometry.estimators import LDAEstimator
from llmgeometry.interventions import steer_at_layer
from llmgeometry.loaders import load_geometry, load_teacher_vectors
from llmgeometry.validation import sibling_ratio_kl_from_logits


def capture_last_token_by_layer(model, tokenizer, prompts: List[str], layers: List[int], device: str, max_length: int = 64):
    tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**tok, output_hidden_states=True)
        hs = out.hidden_states  # tuple length = n_layers+1 (incl embeddings)
    # Map negative indices to python indices over module blocks: assume hs[1:] match blocks
    n_blocks = len(hs) - 1
    out_map: Dict[int, torch.Tensor] = {}
    last_idx = tok["attention_mask"].sum(dim=1) - 1
    for li in layers:
        idx = li if li >= 0 else (n_blocks + li)
        if idx < 0 or idx >= n_blocks:
            continue
        H = hs[idx + 1].detach().float()  # [B, T, d]
        last = H[torch.arange(H.size(0)), last_idx]  # [B, d]
        out_map[int(li)] = last.cpu()
    return out_map


def child_token_ids(tok, name: str) -> List[int]:
    ids = tok(" " + name, add_special_tokens=False).input_ids
    out = []
    if ids:
        out.append(ids[-1])
    if not name.endswith("s"):
        ids2 = tok(" " + name + "s", add_special_tokens=False).input_ids
        if ids2:
            out.append(ids2[-1])
    return sorted(list(set(out)))


def main():
    ap = argparse.ArgumentParser(description="Exp10b: Emergence curves")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp10b"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("run", {}).get("device", "cpu")
    model_name = cfg["model"]["name"]
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    geom = load_geometry(cfg["inputs"]["geometry"])  # fixed geometry from Exp01
    parents, child_vecs, child_deltas = load_teacher_vectors(cfg["inputs"]["teacher_vectors"])
    hier = json.load(open(cfg["inputs"]["hierarchies_json"]))

    layers = [int(x) for x in cfg.get("eval", {}).get("layers", [-1, -2, -3, -6])]
    alpha = float(cfg.get("eval", {}).get("magnitude", 0.5))
    k_parent = int(cfg.get("data", {}).get("prompts_per_parent", 8))
    k_child = int(cfg.get("data", {}).get("prompts_per_child", 4))

    # Build prompt + token sets
    parent_prompts: Dict[str, List[str]] = {}
    child_prompts: Dict[str, Dict[str, List[str]]] = {}
    sibling_ids: Dict[str, Dict[str, List[int]]] = {}
    for h in hier:
        pid = h["parent"]["synset_id"]
        parent_prompts[pid] = h.get("parent_prompts", [])[:k_parent]
        cps = {}
        sids = {}
        for c in h.get("children", []):
            cid = c.get("synset_id")
            nm = c.get("name", cid.split(".")[0])
            ps = h.get("child_prompts", {}).get(cid, [])[:k_child]
            if ps:
                cps[cid] = ps
            sids[cid] = child_token_ids(tok, nm)
        child_prompts[pid] = cps
        sibling_ids[pid] = sids

    # Angle vs layer: per child compare LDA@layer vs teacher delta
    lda = LDAEstimator(shrinkage=float(cfg.get("geometry", {}).get("lda_shrinkage", 0.1)))
    angle_vs_layer: Dict[int, List[float]] = {int(li): [] for li in layers}
    for pid, cps in child_prompts.items():
        for cid, ps in cps.items():
            # negatives = sibling prompts under same parent
            neg = []
            for ocid, ops in cps.items():
                if ocid != cid:
                    neg.extend(ops[:max(1, len(ps))])
            if not ps or not neg:
                continue
            acts_map = capture_last_token_by_layer(mdl, tok, ps + neg, layers, device)
            n_pos = len(ps)
            for li, X in acts_map.items():
                X_pos = X[:n_pos]
                X_neg = X[n_pos:]
                w = lda.estimate_binary_direction(X_pos, X_neg, geom, normalize=True)
                delt = child_deltas.get(pid, {}).get(cid)
                if delt is None:
                    continue
                ang = torch.rad2deg(geom.causal_angle(w, delt)).item()
                angle_vs_layer[int(li)].append(float(ang))

    angle_curve = {str(int(li)): (float(np.median(v)) if v else float("nan")) for li, v in angle_vs_layer.items()}

    # KL vs layer: edits at each layer under fixed geometry
    kl_vs_layer: Dict[int, List[float]] = {int(li): [] for li in layers}
    for pid, prompts in parent_prompts.items():
        if not prompts:
            continue
        pvec = torch.tensor(parents.get(pid, None), dtype=torch.float32)
        if pvec is None:
            continue
        v = geom.normalize_causal(pvec)
        sib = sibling_ids.get(pid, {})
        for li in layers:
            try:
                out = steer_at_layer(mdl, tok, prompts, v, magnitude=alpha, layer_index=int(li), device=device)
            except Exception:
                continue
            if out["baseline_logits"].shape != out["steered_logits"].shape:
                continue
            kl_list = []
            for ids in sib.values():
                if not ids:
                    continue
                kl = sibling_ratio_kl_from_logits(out["baseline_logits"], out["steered_logits"], ids)
                kl_list.append(kl)
            if kl_list:
                kl_vs_layer[int(li)].append(float(np.median(kl_list)))

    kl_curve = {str(int(li)): (float(np.median(v)) if v else float("nan")) for li, v in kl_vs_layer.items()}

    out = {
        "angle_vs_layer": angle_curve,
        "kl_vs_layer": kl_curve,
        "n_angle_per_layer": {str(int(li)): len(angle_vs_layer[int(li)]) for li in layers},
        "n_kl_per_layer": {str(int(li)): len(kl_vs_layer[int(li)]) for li in layers},
    }
    with open(out_dir / "emergence_curves.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", out_dir / "emergence_curves.json")


if __name__ == "__main__":
    main()
