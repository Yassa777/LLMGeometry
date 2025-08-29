#!/usr/bin/env python3
"""
Exp08: Dataset variants — compare angle medians across multiple teacher_vectors.json files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import yaml

from llmgeometry.loaders import save_json
from typing import Dict
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    _WN_OK = True
except Exception:
    _WN_OK = False


def median_angle_from_tv(path: str) -> float:
    data = json.load(open(path))
    st = data.get("angle_stats", {})
    return float(st.get("median_angle_deg", float("nan")))


def polysemy_subset_from_tv(path: str) -> Dict[str, float]:
    try:
        data = json.load(open(path))
    except Exception:
        return {}
    pairs = data.get("angle_pairs", [])
    if not pairs or not _WN_OK:
        return {}
    mono, poly = [], []
    for item in pairs:
        cid = item.get("child_id")
        ang = item.get("angle_deg")
        if cid is None or ang is None:
            continue
        if "." in cid:
            try:
                ss = wn.synset(cid)
                # polysemy as number of lemmas across senses for lemma string
                lemmas = {l.name() for l in ss.lemmas()}
                polysemous = any(len(wn.synsets(lm.split("_")[0])) > 1 for lm in lemmas)
                if polysemous:
                    poly.append(float(ang))
                else:
                    mono.append(float(ang))
            except Exception:
                continue
    out: Dict[str, float] = {}
    if mono:
        out["monosemous_median"] = float(np.median(mono))
    if poly:
        out["polysemous_median"] = float(np.median(poly))
    if mono and poly:
        out["delta_poly_minus_mono"] = float(np.median(poly) - np.median(mono))
    return out


def main():
    ap = argparse.ArgumentParser(description="Exp08: Dataset variants")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp08"))
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: List[str] = cfg.get("inputs", {}).get("teacher_vectors_list", [])
    out = {}
    for p in variants:
        med = median_angle_from_tv(p)
        entry = {"angle_median_deg": med}
        poly = polysemy_subset_from_tv(p)
        if poly:
            entry.update({"polysemy": poly})
        out[str(p)] = entry
    # Summary
    meds = [v["angle_median_deg"] for v in out.values() if not np.isnan(v["angle_median_deg"])]
    summary = {"median_of_medians": float(np.median(meds)) if meds else float("nan")}
    save_json({"per_variant": out, "summary": summary}, str(out_dir / "dataset_variants.json"))
    print("Saved:", out_dir / "dataset_variants.json")


if __name__ == "__main__":
    main()
