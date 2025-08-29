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


def median_angle_from_tv(path: str) -> float:
    data = json.load(open(path))
    st = data.get("angle_stats", {})
    return float(st.get("median_angle_deg", float("nan")))


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
        out[str(p)] = {"angle_median_deg": med}
    # Summary
    meds = [v["angle_median_deg"] for v in out.values() if not np.isnan(v["angle_median_deg"])]
    summary = {"median_of_medians": float(np.median(meds)) if meds else float("nan")}
    save_json({"per_variant": out, "summary": summary}, str(out_dir / "dataset_variants.json"))
    print("Saved:", out_dir / "dataset_variants.json")


if __name__ == "__main__":
    main()

