#!/usr/bin/env python3
"""
Exp02b: Ratio-invariance from logits under parent edits with controls.

For each parent:
 - Build sibling token-id sets from child names (last sub-token id per name, incl. simple plural alias)
 - Collect baseline last-token logits for parent prompts
 - Apply edits with α∈{-1.0,-0.5,0,0.5,1.0} along parent vector
 - Compute KL(base||after) over sibling distributions and ΔΔ bars (parent vs siblings)

Controls: Euclid norm, Random-Parent replacement, Shuffled-U geometry.
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
from llmgeometry.estimators import LDAEstimator
from llmgeometry.geometry import geometry_from_shuffled_unembedding
from llmgeometry.loaders import load_geometry, load_teacher_vectors, load_activations
from llmgeometry.validation import sibling_ratio_kl_from_logits


def child_token_ids_from_names(tok, names: List[str]) -> List[int]:
    out: List[int] = []
    for nm in names:
        # get last sub-token id of the string (with leading space for GPT2-like)
        ids = tok(" " + nm, add_special_tokens=False).input_ids
        if ids:
            out.append(ids[-1])
        # crude plural alias
        if not nm.endswith("s"):
            ids2 = tok(" " + nm + "s", add_special_tokens=False).input_ids
            if ids2:
                out.append(ids2[-1])
    return sorted(list(set(out)))


def logits_last_token(logits: torch.Tensor, last_idx: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    if not token_ids:
        return torch.zeros(logits.size(0))
    vals = []
    for i in range(logits.size(0)):
        li = int(last_idx[i].item())
        vals.append(float(logits[i, li, token_ids].mean().item()))
    return torch.tensor(vals, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description="Exp02b: Ratio-invariance (logits)")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp02b"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    device = cfg.get("run", {}).get("device", "cpu")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    geom = load_geometry(cfg["inputs"]["geometry"])  # baseline causal geometry
    parents, _, _ = load_teacher_vectors(cfg["inputs"]["teacher_vectors"])
    hier = json.load(open(cfg["inputs"]["hierarchies_json"]))

    # sibling token sets & parent prompts
    sibling_ids: Dict[str, Dict[str, List[int]]] = {}
    parent_prompts: Dict[str, List[str]] = {}
    for h in hier:
        pid = h["parent"]["synset_id"]
        parent_prompts[pid] = h.get("parent_prompts", [])[: int(cfg.get("data", {}).get("prompts_per_parent", 8))]
        sib = {}
        # Prefer provided sibling_tokens if present (aliases)
        provided = h.get("sibling_tokens", {})
        if provided:
            for cid, alias_list in provided.items():
                sib[cid] = child_token_ids_from_names(tok, alias_list)
        else:
            # Fallback to child names
            for c in h.get("children", []):
                cid = c.get("synset_id")
                nm = c.get("name", cid.split(".")[0])
                sib[cid] = child_token_ids_from_names(tok, [nm])
        sibling_ids[pid] = sib

    alphas = cfg.get("eval", {}).get("alphas", [-1.0, -0.5, 0.0, 0.5, 1.0])
    conds = ["causal", "euclid", "random_parent", "shuffled_U"]

    # Shuffled-U geometry
    try:
        U = mdl.get_output_embeddings().weight.detach().to("cpu", dtype=torch.float32)
        geom_shuf = geometry_from_shuffled_unembedding(U, shrinkage=geom.shrinkage, seed=0)
    except Exception:
        geom_shuf = None

    pids = [pid for pid in parents.keys() if pid in parent_prompts]
    results: Dict[str, Dict] = {}
    # Optional Euclidean (identity-geometry) parent directions from activations
    euclid_dirs: Dict[str, torch.Tensor] = {}
    acts = None
    try:
        if "activations" in cfg.get("inputs", {}):
            acts = load_activations(cfg["inputs"]["activations"])  # {concept_id: {pos,neg}}
    except Exception:
        acts = None

    # Find hidden size d from a single forward
    d_hidden = None
    try:
        with torch.no_grad():
            _tmp_inputs = tok(["test"], return_tensors="pt").to(device)
            _out = mdl(**_tmp_inputs, output_hidden_states=True)
            d_hidden = int(_out.hidden_states[-1].shape[-1])
    except Exception:
        pass

    if acts is not None and d_hidden is not None:
        id_geom = CausalGeometry(torch.eye(d_hidden), shrinkage=0.0)
        lda_id = LDAEstimator(shrinkage=0.0)
        for pid in pids:
            a = acts.get(pid)
            if a and "pos" in a and "neg" in a and len(a["pos"]) and len(a["neg"]):
                try:
                    w = lda_id.estimate_binary_direction(a["pos"], a["neg"], id_geom, normalize=True)
                    euclid_dirs[pid] = w.to(torch.float32)
                except Exception:
                    continue

    for pid in pids:
        prompts = parent_prompts[pid]
        sib = sibling_ids.get(pid, {})
        if not prompts or not sib:
            continue
        pvec = torch.tensor(parents[pid], dtype=torch.float32)
        other_pids = [q for q in pids if q != pid]
        rp_vec = torch.tensor(parents[random.choice(other_pids)], dtype=torch.float32) if other_pids else pvec

        per_cond = {}
        with torch.no_grad():
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            out0 = mdl(**inputs)
            logits0 = out0.logits.detach().cpu()
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
        for cond in conds:
            by_a = {}
            for a in alphas:
                # choose editing direction per condition
                if cond == "causal":
                    v = geom.normalize_causal(pvec)
                elif cond == "euclid":
                    # Prefer Euclidean LDA direction under identity geometry if available
                    if pid in euclid_dirs:
                        v = euclid_dirs[pid] / (euclid_dirs[pid].norm() + 1e-8)
                    else:
                        v = pvec / (pvec.norm() + 1e-8)
                elif cond == "random_parent":
                    v = geom.normalize_causal(rp_vec)
                elif cond == "shuffled_U" and geom_shuf is not None:
                    v = geom_shuf.normalize_causal(pvec)
                else:
                    v = geom.normalize_causal(pvec)
                with torch.no_grad():
                    outputs = mdl(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1].clone()
                    li = inputs["attention_mask"].sum(dim=1) - 1
                    vv = v.to(hidden.dtype).to(device)
                    hidden[torch.arange(hidden.size(0)), li] += float(a) * vv
                    head = mdl.lm_head if hasattr(mdl, "lm_head") else mdl.get_output_embeddings()
                    logits1 = head(hidden).detach().cpu()

                # sibling KLs across children
                kls = []
                for ids in sib.values():
                    if not ids:
                        continue
                    kl = sibling_ratio_kl_from_logits(logits0, logits1, ids)
                    kls.append(kl)
                med_kl = float(np.median(kls)) if kls else float("nan")

                # ΔΔ parent vs siblings: parent tokens = union of children under this parent
                all_ids = sorted(list(set([x for ids in sib.values() for x in ids])))
                base_p = logits_last_token(logits0, last_idx, all_ids)
                edit_p = logits_last_token(logits1, last_idx, all_ids)
                dparent = float((edit_p - base_p).mean().item())
                # child |ΔΔ|
                child_dd = []
                for cid, ids in sib.items():
                    if not ids:
                        continue
                    base_c = logits_last_token(logits0, last_idx, ids)
                    edit_c = logits_last_token(logits1, last_idx, ids)
                    others = sorted(list(set(all_ids) - set(ids))) or ids
                    base_s = logits_last_token(logits0, last_idx, others)
                    edit_s = logits_last_token(logits1, last_idx, others)
                    ddelta = (edit_c - edit_s) - (base_c - base_s)
                    child_dd.append(float(torch.abs(ddelta).mean().item()))
                child_dd_med = float(np.median(child_dd)) if child_dd else float("nan")

                by_a[str(a)] = {"median_kl": med_kl, "parent_ddelta": dparent, "child_ddelta_abs": child_dd_med}
            per_cond[cond] = by_a
        results[pid] = per_cond

    # Global summaries
    summary = {}
    for cond in conds:
        ks, pd, cd = [], [], []
        for pid in results:
            for a, d in results[pid][cond].items():
                if float(a) == 0.0:
                    continue
                if np.isfinite(d.get("median_kl", np.nan)):
                    ks.append(d["median_kl"])
                if np.isfinite(d.get("parent_ddelta", np.nan)):
                    pd.append(d["parent_ddelta"])
                if np.isfinite(d.get("child_ddelta_abs", np.nan)):
                    cd.append(d["child_ddelta_abs"])
        summary[cond] = {
            "median_kl": float(np.median(ks)) if ks else float("nan"),
            "median_parent_ddelta": float(np.median(pd)) if pd else float("nan"),
            "median_child_ddelta_abs": float(np.median(cd)) if cd else float("nan"),
            "n": int(len(ks)),
        }

    with open(out_dir / "ratio_invariance_logits.json", "w") as f:
        json.dump({"per_parent": results, "summary": summary}, f, indent=2)
    print("Saved:", out_dir / "ratio_invariance_logits.json")


if __name__ == "__main__":
    main()
