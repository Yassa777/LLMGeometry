#!/usr/bin/env python3
"""
Build a hierarchical concept dataset from WordNet with curated children and prompts.

Features:
- Select parent synsets (default across diverse domains).
- Choose top-N child hyponyms per parent using word frequency (Zipf) filters.
- Generate diverse prompts per concept using templates and synset definitions.

Usage:
  python tools/build_wordnet_hierarchy.py \
    --out runs/exp01/concept_hierarchies.json \
    --children-per-parent 6 \
    --prompts-per-concept 24 \
    --min-zipf 3.0

Optionally specify parents explicitly:
  --parents animal.n.01,vehicle.n.01,profession.n.01,food.n.01,emotion.n.01,programming_language.n.01,geographical_area.n.01,art.n.01
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import nltk
from wordfreq import zipf_frequency

try:
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover
    wn = None


DEFAULT_PARENTS = [
    "animal.n.01",
    "vehicle.n.01",
    "profession.n.01",
    "food.n.01",
    "emotion.n.01",
    "programming_language.n.01",
    "geographical_area.n.01",
    "art.n.01",
]


def ensure_wordnet():
    global wn
    try:
        if wn is None:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet as _wn
            wn = _wn
        else:
            # Probe
            _ = wn.synsets("dog")
    except Exception:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet as _wn
        wn = _wn


def best_lemma_name(ss, min_zipf: float) -> Tuple[str, float]:
    best = (None, float("-inf"))
    for l in ss.lemmas():
        name = l.name().replace("_", " ")
        # Prefer simple tokens
        if not re.match(r"^[A-Za-z][a-z]+(?: [A-Za-z][a-z]+)?$", name):
            continue
        z = zipf_frequency(name, "en")
        if z >= min_zipf and z > best[1]:
            best = (name, z)
    if best[0] is None:
        # fallback: first lemma stripped underscores
        if ss.lemmas():
            nm = ss.lemmas()[0].name().replace("_", " ")
            return nm, zipf_frequency(nm, "en")
        return ss.name(), float("-inf")
    return best


def gather_children(parent_ss, k: int, min_zipf: float) -> List[Tuple[str, str, float]]:
    # collect immediate hyponyms, extend to depth-2 if needed
    hyps = list(parent_ss.hyponyms())
    if len(hyps) < k:
        for h in list(hyps):
            hyps.extend(h.hyponyms())
    seen = set()
    scored: List[Tuple[str, str, float]] = []  # (synset_id, display_name, zipf)
    for ss in hyps:
        if ss.pos() != "n":
            continue
        name, z = best_lemma_name(ss, min_zipf)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        scored.append((ss.name(), name, z))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:k]


def generate_prompts(name: str, definition: str, n: int) -> List[str]:
    base = [
        f"A sentence about {name}.",
        f"Describe {name}.",
        f"A short fact about {name}.",
        f"An observation about {name}.",
        f"Explain what is typical of {name}.",
        f"Mention a common use or role of {name}.",
        f"One property often seen in {name}.",
        f"Give an everyday example involving {name}.",
    ]
    # Definition-based, avoids repeating the lemma
    def_snip = definition[:120] if definition else "the concept"
    base += [
        f"Write one sentence relevant to: {def_snip}.",
        f"Compose a sentence that reflects: {def_snip}.",
    ]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Build WordNet-backed concept hierarchies")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--parents", type=str, default=",".join(DEFAULT_PARENTS), help="Comma-separated synset IDs")
    ap.add_argument("--children-per-parent", type=int, default=6)
    ap.add_argument("--prompts-per-concept", type=int, default=24)
    ap.add_argument("--min-zipf", type=float, default=3.0, help="Min Zipf frequency for lemma selection")
    args = ap.parse_args()

    ensure_wordnet()

    parent_ids = [p.strip() for p in args.parents.split(",") if p.strip()]
    data: List[Dict] = []
    for pid in parent_ids:
        try:
            pss = wn.synset(pid)
        except Exception:
            print(f"Warning: could not resolve parent synset {pid}; skipping")
            continue
        p_name, _ = best_lemma_name(pss, args.min_zipf)
        p_prompts = generate_prompts(p_name, pss.definition(), args.prompts_per_concept)
        children = gather_children(pss, args.children_per_parent, args.min_zipf)
        child_prompts: Dict[str, List[str]] = {}
        child_nodes: List[Dict] = []
        for ssid, disp, _ in children:
            try:
                ss = wn.synset(ssid)
            except Exception:
                continue
            child_nodes.append({"synset_id": ss.name(), "name": disp, "prompts": []})
            child_prompts[ss.name()] = generate_prompts(disp, ss.definition(), args.prompts_per_concept)
        if not child_nodes:
            print(f"Warning: no children selected for {pid}; skipping")
            continue
        data.append(
            {
                "parent": {"synset_id": pss.name(), "name": p_name, "prompts": []},
                "children": child_nodes,
                "parent_prompts": p_prompts,
                "child_prompts": child_prompts,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Wrote WordNet hierarchy:", out_path)


if __name__ == "__main__":
    main()

