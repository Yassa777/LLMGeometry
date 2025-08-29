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
                # Tokenizer helpers: last subtoken id with leading space
                def token_id_list(strings):
                    ids = []
                    for s in strings:
                        toks = tok(" " + s, add_special_tokens=False).input_ids
                        if toks:
                            ids.append(toks[-1])
                    # dedup
                    return sorted(list(set(ids)))
                # Build parent token id union from sibling_tokens aliases
                sib_alias = hh.get("sibling_tokens", {})
                parent_token_ids = token_id_list([a for aliases in sib_alias.values() for a in aliases])
                # Build child-specific ids from aliases
                child_token_ids = {cid: token_id_list(aliases) for cid, aliases in sib_alias.items()}
                for m in magnitudes:
                    out = steer_parent_vector(mdl, tok, prompts, pvec, magnitude=float(m), device=device)
                    logits0 = out["baseline_logits"].float()
                    logits1 = out["steered_logits"].float()
                    # Parent ΔΔ: union over all child aliases
                    if parent_token_ids:
                        da = logits1[:, -1, parent_token_ids].mean(dim=-1)
                        db = logits0[:, -1, parent_token_ids].mean(dim=-1)
                        dparent = float((da - db).mean().item())
                        locality["parent_ddelta"].setdefault(str(float(m)), []).append(dparent)
                    # Child |ΔΔ| vs siblings
                    for cid, ids in child_token_ids.items():
                        if not ids:
                            continue
                        others = sorted(list(set(parent_token_ids) - set(ids))) or ids
                        da = logits1[:, -1, ids].mean(dim=-1) - logits1[:, -1, others].mean(dim=-1)
                        db = logits0[:, -1, ids].mean(dim=-1) - logits0[:, -1, others].mean(dim=-1)
                        ddelta = torch.abs(da - db).mean()
                        locality["child_ddelta_abs"].setdefault(str(float(m)), []).append(float(ddelta.item()))
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
