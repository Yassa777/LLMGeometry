#!/usr/bin/env python3
"""
Exp09: Token granularity comparison.

Given two teacher_vectors.json files representing different token granularities
(e.g., last-token vs pooled variants), compare angle medians.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from llmgeometry.loaders import save_json
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    from wordfreq import zipf_frequency  # type: ignore
    _FREQ_OK = True
except Exception:
    _FREQ_OK = False


def median_from_tv(path: str) -> float:
    data = json.load(open(path))
    return float(data.get("angle_stats", {}).get("median_angle_deg", float("nan")))


def main():
    ap = argparse.ArgumentParser(description="Exp09: Token granularity compare")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp09"))
    out_dir.mkdir(parents=True, exist_ok=True)

    last_token = cfg["inputs"].get("last_token_tv")
    pooled = cfg["inputs"].get("pooled_tv")

    m_last = median_from_tv(last_token) if last_token else float("nan")
    m_pool = median_from_tv(pooled) if pooled else float("nan")

    # Optional: angle vs log-freq curve using angle_pairs and WordNet IDs
    freq_curve = {}
    try:
        if _FREQ_OK and last_token:
            data = json.load(open(last_token))
            pairs = data.get("angle_pairs", [])
            if pairs:
                xs, ys = [], []
                for item in pairs:
                    cid = item.get("child_id")
                    ang = item.get("angle_deg")
                    if not cid or ang is None:
                        continue
                    if "." in cid:
                        try:
                            ss = wn.synset(cid)
                            # choose primary lemma
                            lemma = ss.lemmas()[0].name().replace("_", " ")
                            z = zipf_frequency(lemma, "en")
                            if np.isfinite(z):
                                xs.append(z)
                                ys.append(float(ang))
                        except Exception:
                            continue
                # Bin by Zipf frequency
                if xs:
                    xs = np.array(xs)
                    ys = np.array(ys)
                    bins = np.linspace(1.5, 7.0, 12)
                    mids = 0.5 * (bins[1:] + bins[:-1])
                    med = []
                    for i in range(len(bins) - 1):
                        mask = (xs >= bins[i]) & (xs < bins[i + 1])
                        if np.any(mask):
                            med.append(float(np.median(ys[mask])))
                        else:
                            med.append(float("nan"))
                    freq_curve = {str(float(m)): float(v) for m, v in zip(mids, med)}
    except Exception:
        freq_curve = {}

    out = {
        "last_token": {"angle_median_deg": m_last},
        "pooled": {"angle_median_deg": m_pool},
        "delta": (float(m_pool - m_last) if (not np.isnan(m_last) and not np.isnan(m_pool)) else float("nan")),
    }
    if freq_curve:
        out["angle_vs_logfreq"] = freq_curve
    save_json(out, str(out_dir / "token_granularity.json"))
    print("Saved:", out_dir / "token_granularity.json")


if __name__ == "__main__":
    main()
