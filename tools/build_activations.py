#!/usr/bin/env python3
"""
Build hierarchical activations (HDF5) from a concept hierarchy JSON.

For each concept id (parent and children), collects last-token activations for
positive prompts (provided) and a simple negative set derived from siblings or
other concepts. Saves to HDF5 expected by experiments (pos/neg per concept).

Usage:
  python tools/build_activations.py \
    --model distilgpt2 \
    --hier llmgeometry-repo/runs/exp01/concept_hierarchies.json \
    --out  llmgeometry-repo/runs/exp01/activations.h5 \
    --device cpu --max-pos 8 --max-neg 8 --granularity last

Notes:
  - Negatives are heuristically chosen:
    * Parent concept: neg = prompts from this parent's children
      (fallback: prompts from other parents if no children)
    * Child concept: neg = prompts from sibling children under the same parent
      (fallback: parent prompts if no siblings)
  - We cap pos/neg counts per concept for speed via --max-pos/--max-neg.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from pathlib import Path

import torch

from llmgeometry.activations import ActivationCapture, ActivationConfig
from llmgeometry.concepts import load_concept_hierarchies


def choose_negatives(hier, max_neg: int) -> Dict[str, List[str]]:
    """Return mapping concept_id -> negative prompts list (heuristic)."""
    negs: Dict[str, List[str]] = {}
    # Build helper maps
    parent_ids = [h.parent.synset_id for h in hier]
    parent_prompts = {h.parent.synset_id: list(h.parent_prompts) for h in hier}
    child_prompts = {}
    for h in hier:
        for c in h.children:
            cid = c.synset_id
            child_prompts[cid] = list(h.child_prompts.get(cid, []))

    # Parents: neg from own children, else other parents
    for h in hier:
        pid = h.parent.synset_id
        neg_pool: List[str] = []
        for c in h.children:
            neg_pool.extend(h.child_prompts.get(c.synset_id, []))
        if not neg_pool:
            # fallback to other parents' prompts
            for qpid, ps in parent_prompts.items():
                if qpid != pid:
                    neg_pool.extend(ps)
        negs[pid] = neg_pool[:max_neg] if max_neg is not None else neg_pool

    # Children: neg from siblings, else parent prompts
    for h in hier:
        pid = h.parent.synset_id
        sib_map = {c.synset_id: list(h.child_prompts.get(c.synset_id, [])) for c in h.children}
        for cid in sib_map.keys():
            neg_pool = []
            for ocid, ps in sib_map.items():
                if ocid != cid:
                    neg_pool.extend(ps)
            if not neg_pool:
                neg_pool = list(parent_prompts.get(pid, []))
            negs[cid] = neg_pool[:max_neg] if max_neg is not None else neg_pool

    return negs


def main():
    ap = argparse.ArgumentParser(description="Build hierarchical activations HDF5 from hierarchy JSON")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--hier", type=str, required=True, help="Path to concept_hierarchies.json")
    ap.add_argument("--out", type=str, required=True, help="Output HDF5 path")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-pos", type=int, default=8, help="Max positive prompts per concept (truncate)")
    ap.add_argument("--max-neg", type=int, default=8, help="Max negative prompts per concept (truncate)")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--granularity", type=str, default="last", choices=["last", "pooled"], help="Token granularity for activations")
    args = ap.parse_args()

    hier = load_concept_hierarchies(args.hier)
    # Build pos prompts map
    pos_prompts: Dict[str, List[str]] = {}
    for h in hier:
        pid = h.parent.synset_id
        pos_prompts[pid] = list(h.parent_prompts)[: args.max_pos] if args.max_pos is not None else list(h.parent_prompts)
        for c in h.children:
            cid = c.synset_id
            cps = h.child_prompts.get(cid, [])
            pos_prompts[cid] = cps[: args.max_pos] if args.max_pos is not None else list(cps)

    neg_prompts = choose_negatives(hier, args.max_neg)

    # Capture activations
    # Choose dtype based on device
    import torch
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    cap = ActivationCapture(ActivationConfig(model_name=args.model, device=args.device, max_length=args.max_length, dtype=dtype))

    # Assemble concept dict
    concept_acts: Dict[str, Dict[str, torch.Tensor]] = {}
    for cid, pps in pos_prompts.items():
        if not pps:
            continue
        nps = neg_prompts.get(cid, [])
        if args.granularity == "pooled":
            pos_acts = cap.capture_pooled_activations(pps)
            neg_acts = cap.capture_pooled_activations(nps) if nps else torch.empty(0, pos_acts.size(-1))
        else:
            pos_acts = cap.capture_last_token_activations(pps)
            neg_acts = cap.capture_last_token_activations(nps) if nps else torch.empty(0, pos_acts.size(-1))
        concept_acts[cid] = {"pos": pos_acts, "neg": neg_acts}

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ActivationCapture.save_hierarchical_activations(concept_acts, str(out_path))
    print("Wrote activations:", out_path)


if __name__ == "__main__":
    main()
