#!/usr/bin/env python3
"""
Exp05b: Estimator shoot-out — LDA vs class-mean vs L2-probe.

For each concept with pos/neg activations, estimate directions using
 - LDAEstimator
 - MeanDiffEstimator
 - L2ProbeEstimator (logistic with L2)
Compute AUROC and angles to teacher vectors if provided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from llmgeometry import CausalGeometry
from llmgeometry.loaders import load_geometry, load_activations, save_json
from llmgeometry import steer_parent_vector
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmgeometry.estimators import LDAEstimator, MeanDiffEstimator, L2ProbeEstimator


def score_auc(direction: torch.Tensor, X_pos: torch.Tensor, X_neg: torch.Tensor, geom: CausalGeometry) -> float:
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
    ap = argparse.ArgumentParser(description="Exp05b: Estimator shoot-out")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp05b"))
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = load_geometry(cfg["inputs"]["geometry"])  # geometry.pt from Exp01
    acts = load_activations(cfg["inputs"]["activations"])  # HDF5
    teacher_vecs = None
    if "teacher_vectors" in cfg.get("inputs", {}):
        tv = json.load(open(cfg["inputs"]["teacher_vectors"]))
        teacher_vecs = tv.get("parent_vectors", {})

    lda = LDAEstimator(shrinkage=float(cfg.get("geometry", {}).get("lda_shrinkage", 0.1)))
    md = MeanDiffEstimator()
    l2 = L2ProbeEstimator(C=float(cfg.get("probe", {}).get("C", 1.0)), max_iter=int(cfg.get("probe", {}).get("max_iter", 500)))

    # Optional off-target ΔΔ setup
    hier = None
    try:
        if "hierarchies_json" in cfg.get("inputs", {}):
            import json as _json
            hier = _json.load(open(cfg["inputs"]["hierarchies_json"]))
    except Exception:
        hier = None
    model_name = cfg.get("model", {}).get("name")
    device = cfg.get("run", {}).get("device", "cpu")
    tok = None
    mdl = None
    if hier and model_name:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    # Build child->parent and unrelated token sets (union of tokens from other parents)
    child_to_parent: Dict[str, str] = {}
    parent_prompts = {}
    other_parent_tokens: Dict[str, List[int]] = {}
    if hier and tok:
        # token ids per parent (union of children's names)
        parent_child_tokens: Dict[str, List[int]] = {}
        for h in hier:
            pid = h["parent"]["synset_id"]
            parent_prompts[pid] = h.get("parent_prompts", [])[:2]
            ids = []
            for c in h.get("children", []):
                nm = c.get("name", c.get("synset_id", "").split(".")[0])
                i = tok(" " + nm, add_special_tokens=False).input_ids
                if i:
                    ids.append(i[-1])
                if not nm.endswith("s"):
                    i2 = tok(" " + nm + "s", add_special_tokens=False).input_ids
                    if i2:
                        ids.append(i2[-1])
                child_to_parent[c.get("synset_id")] = pid
            parent_child_tokens[pid] = sorted(list(set(ids)))
        for pid in parent_child_tokens:
            others = []
            for qid, ids in parent_child_tokens.items():
                if qid != pid:
                    others.extend(ids)
            other_parent_tokens[pid] = sorted(list(set(others)))

    per: Dict[str, Dict[str, float]] = {}
    auc_lda = []
    auc_md = []
    auc_l2 = []
    for cid, d in acts.items():
        if "pos" not in d or "neg" not in d or len(d["pos"]) == 0 or len(d["neg"]) == 0:
            continue
        w_lda = lda.estimate_binary_direction(d["pos"], d["neg"], geom, normalize=True)
        w_md = md.estimate_binary_direction(d["pos"], d["neg"], geom, normalize=True)
        w_l2 = l2.estimate_binary_direction(d["pos"], d["neg"], geom, normalize=True)
        a1 = score_auc(w_lda, d["pos"], d["neg"], geom)
        a2 = score_auc(w_md, d["pos"], d["neg"], geom)
        a3 = score_auc(w_l2, d["pos"], d["neg"], geom)
        # Within-class variance along direction (whitened projection)
        def within_vars(w):
            ww = geom.normalize_causal(w.to(torch.float32))
            ww_w = geom.whiten(ww)
            xp = geom.whiten(d["pos"].to(torch.float32))
            xn = geom.whiten(d["neg"].to(torch.float32))
            sp = (xp * ww_w).sum(dim=-1).var().item() if len(xp) > 1 else float("nan")
            sn = (xn * ww_w).sum(dim=-1).var().item() if len(xn) > 1 else float("nan")
            return sp, sn
        v1p, v1n = within_vars(w_lda)
        v2p, v2n = within_vars(w_md)
        v3p, v3n = within_vars(w_l2)

        res = {
            "auroc_lda": a1, "auroc_mean_diff": a2, "auroc_l2probe": a3,
            "var_pos_lda": v1p, "var_neg_lda": v1n,
            "var_pos_mean_diff": v2p, "var_neg_mean_diff": v2n,
            "var_pos_l2probe": v3p, "var_neg_l2probe": v3n,
        }
        if teacher_vecs and cid in teacher_vecs:
            t = torch.tensor(teacher_vecs[cid], dtype=torch.float32)
            res.update({
                "angle_to_teacher_lda": float(torch.rad2deg(geom.causal_angle(w_lda, t)).item()),
                "angle_to_teacher_mean_diff": float(torch.rad2deg(geom.causal_angle(w_md, t)).item()),
                "angle_to_teacher_l2probe": float(torch.rad2deg(geom.causal_angle(w_l2, t)).item()),
            })
        # Off-target ΔΔ using unrelated contrasts
        if hier and mdl and tok:
            pid = child_to_parent.get(cid)
            if pid and parent_prompts.get(pid) and other_parent_tokens.get(pid):
                prompts = parent_prompts[pid]
                tokens = other_parent_tokens[pid]
                def offtarget(w):
                    out = steer_parent_vector(mdl, tok, prompts, w, magnitude=float(cfg.get("eval", {}).get("magnitude", 0.5)), device=device)
                    logits0 = out["baseline_logits"].float()
                    logits1 = out["steered_logits"].float()
                    # last-token Δ on unrelated tokens per-prompt
                    B, T, V = logits0.shape
                    lt = logits0[:, -1, tokens]
                    lt1 = logits1[:, -1, tokens]
                    d = (lt1 - lt).abs().mean(dim=-1)
                    return [float(x) for x in d]
                res["offtarget_ddelta_abs_lda"] = offtarget(w_lda)
                res["offtarget_ddelta_abs_mean_diff"] = offtarget(w_md)
                res["offtarget_ddelta_abs_l2probe"] = offtarget(w_l2)
        per[cid] = res
        if np.isfinite(a1): auc_lda.append(a1)
        if np.isfinite(a2): auc_md.append(a2)
        if np.isfinite(a3): auc_l2.append(a3)

    summary = {
        "median_auroc_lda": float(np.median(auc_lda)) if auc_lda else float("nan"),
        "median_auroc_mean_diff": float(np.median(auc_md)) if auc_md else float("nan"),
        "median_auroc_l2probe": float(np.median(auc_l2)) if auc_l2 else float("nan"),
    }
    save_json({"per_concept": per, "summary": summary}, str(out_dir / "estimators.json"))
    print("Saved:", out_dir / "estimators.json")


if __name__ == "__main__":
    main()
