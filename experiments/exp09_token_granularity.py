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

    out = {
        "last_token": {"angle_median_deg": m_last},
        "pooled": {"angle_median_deg": m_pool},
        "delta": (float(m_pool - m_last) if (not np.isnan(m_last) and not np.isnan(m_pool)) else float("nan")),
    }
    save_json(out, str(out_dir / "token_granularity.json"))
    print("Saved:", out_dir / "token_granularity.json")


if __name__ == "__main__":
    main()

