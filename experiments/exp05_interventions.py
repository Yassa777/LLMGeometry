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
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
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
        prompt_mads = {}
        locality = {"parent_ddelta": {}, "child_ddelta_abs": {}}
        for m in magnitudes:
            out = steer_parent_vector(mdl, tok, prompts, pvec, magnitude=float(m), device=device)
            deltas = out["logit_deltas"].float()  # [B, T, V]
            # Re-tokenize to get last indices used (matches steer tokenization)
            tok_batch = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            last_idx = tok_batch["attention_mask"].sum(dim=1) - 1  # [B]
            # Per-prompt mean |Δlogits| at last token
            per_prompt = []
            for i in range(deltas.size(0)):
                li = int(last_idx[i].item())
                per_prompt.append(float(torch.mean(torch.abs(deltas[i, li, :])).item()))
            prompt_mads[str(float(m))] = per_prompt

            mad = torch.mean(torch.abs(deltas))
            per_mag.append({"magnitude": float(m), "mean_abs_delta": float(mad.item())})
            # ΔΔ locality using hierarchy token pairs if available
            # parent_pairs: [[tokA, tokB]] in hierarchy JSON
            # child_pairs: [[child_id1, child_id2]] mapping to tokenization of names
            # Use simple tokenizer on string tokens
            # Load hierarchy for this parent
            # Note: load_hierarchies returns dataclasses; to avoid complexity, parse raw JSON too
        try:
            import json as _json
            hraw = _json.load(open(cfg["inputs"]["hierarchies"]))
            hh = next((h for h in hraw if h["parent"]["synset_id"] == pid), None)
            if hh:
                # Build token pairs
                def tok_id(s):
                    ids = tok(s, add_special_tokens=False).input_ids
                    return ids[0] if ids else None
                for m in magnitudes:
                    out = steer_parent_vector(mdl, tok, prompts, pvec, magnitude=float(m), device=device)
                    logits0 = out["baseline_logits"].float()
                    logits1 = out["steered_logits"].float()
                    # Parent pairs ΔΔ
                    for a, b in hh.get("parent_pairs", []):
                        ia, ib = tok_id(a), tok_id(b)
                        if ia is None or ib is None:
                            continue
                        da = (logits1[:, -1, ia] - logits0[:, -1, ia]).mean()
                        db = (logits1[:, -1, ib] - logits0[:, -1, ib]).mean()
                        ddelta = float((da - db).item())
                        locality["parent_ddelta"].setdefault(str(float(m)), []).append(ddelta)
                    # Child pairs |ΔΔ|
                    for c1, c2 in hh.get("child_pairs", []):
                        n1 = hh.get("child_prompts", {}).get(c1, [c1.split(".")[0]])[0]
                        n2 = hh.get("child_prompts", {}).get(c2, [c2.split(".")[0]])[0]
                        i1, i2 = tok_id(n1), tok_id(n2)
                        if i1 is None or i2 is None:
                            continue
                        d1 = (logits1[:, -1, i1] - logits0[:, -1, i1]).mean()
                        d2 = (logits1[:, -1, i2] - logits0[:, -1, i2]).mean()
                        ddelta = float(torch.abs(d1 - d2).item())
                        locality["child_ddelta_abs"].setdefault(str(float(m)), []).append(ddelta)
        except Exception:
            pass
        # Backward-compatible structure: list for old readers, plus rich fields
        results[pid] = {
            "per_magnitude": per_mag,
            "prompt_mads": prompt_mads,
            "n_prompts": len(prompts),
            "locality": locality,
        }

    with open(out_dir / "interventions.json", "w") as f:
        json.dump({"per_parent": results}, f, indent=2)
    print("Saved:", out_dir / "interventions.json")


if __name__ == "__main__":
    main()
