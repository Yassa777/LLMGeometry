#!/usr/bin/env python3
"""
Quick PASS/FAIL acceptance check across experiment outputs under --base (default: runs).

Criteria (tunable, conservative defaults):
- Exp01: angle median ≥ 80, frac≥80 ≥ 0.7, whiten_fro_error ≤ 1e-2
- Exp02b: causal median KL < 0.10, euclid median KL > 0.25
- Exp03b: median_auroc_lda > median_auroc_mean_diff
- Exp04: boundary-normal median_of_medians ≥ 60 (deg)
- Exp05 locality: parent ΔΔ median > 0, child |ΔΔ| median ≤ 0.15
- Exp06 baseline: frac≥80 ≥ 0.7
- Exp10b: angle and KL curves present (non-NaN)

Usage:
  python tools/check_acceptance.py --base runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(p: Path):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Check acceptance gates")
    ap.add_argument("--base", type=str, default="runs")
    args = ap.parse_args()
    base = Path(args.base)

    rows = []
    def add(name, ok, detail):
        rows.append((name, "PASS" if ok else "FAIL", detail))

    # Exp01
    tv = load_json(base / "exp01" / "teacher_vectors.json")
    if tv:
        ang = tv.get("angle_stats", {})
        geo = tv.get("geometry_stats", {})
        cond = (
            (ang.get("median_angle_deg", 0) >= 80)
            and (ang.get("fraction_above_threshold", 0) >= 0.7)
            and (geo.get("whiten_fro_error", 1) <= 1e-2)
        )
        add("Exp01 geometry & angles", cond, {
            "median": ang.get("median_angle_deg"),
            "frac≥80": ang.get("fraction_above_threshold"),
            "whiten_fro_error": geo.get("whiten_fro_error"),
        })
    else:
        add("Exp01 geometry & angles", False, "missing")

    # Exp02b
    e2b = load_json(base / "exp02b" / "ratio_invariance_logits.json")
    if e2b:
        s = e2b.get("summary", {})
        c = s.get("causal", {})
        e = s.get("euclid", {})
        cond = (
            (c.get("median_kl", 1.0) < 0.10)
            and (e.get("median_kl", 0.0) > 0.25)
        )
        add("Exp02b ratio-invariance (logits)", cond, {
            "causal_median_kl": c.get("median_kl"),
            "euclid_median_kl": e.get("median_kl"),
        })
    else:
        add("Exp02b ratio-invariance (logits)", False, "missing")

    # Exp03b
    e3b = load_json(base / "exp03b" / "contrasts.json")
    if e3b:
        s = e3b.get("summary", {})
        cond = (s.get("median_auroc_lda", 0.0) > s.get("median_auroc_mean_diff", 1.0))
        add("Exp03b contrasts (LDA vs mean-diff)", cond, s)
    else:
        add("Exp03b contrasts", False, "missing")

    # Exp04
    e4 = load_json(base / "exp04" / "boundary_normals.json")
    if e4:
        s = e4.get("summary", {})
        cond = (s.get("median_of_medians", 0.0) >= 60.0)
        add("Exp04 boundary normals", cond, s)
    else:
        add("Exp04 boundary normals", False, "missing")

    # Exp05 locality
    e5 = load_json(base / "exp05" / "interventions.json")
    if e5:
        per = e5.get("per_parent", {})
        # aggregate across parents magnitudes
        pvals, cvals = [], []
        for v in per.values():
            loc = v.get("locality", {})
            for arr in loc.get("parent_ddelta", {}).values():
                pvals.extend(arr)
            for arr in loc.get("child_ddelta_abs", {}).values():
                cvals.extend(arr)
        import numpy as np
        pm = float(np.median(pvals)) if pvals else float("nan")
        cm = float(np.median(cvals)) if cvals else float("nan")
        cond = (pm > 0) and (cm <= 0.15)
        add("Exp05 edit locality", cond, {"parent_ddelta_median": pm, "child_ddelta_abs_median": cm})
    else:
        add("Exp05 edit locality", False, "missing")

    # Exp06
    e6 = load_json(base / "exp06" / "fisher_logit_summary.json")
    if e6:
        b = e6.get("baseline", {})
        cond = (b.get("fraction_above_80", 0.0) >= 0.7)
        add("Exp06 baseline angles", cond, b)
    else:
        add("Exp06 baseline angles", False, "missing")

    # Exp10b
    e10b = load_json(base / "exp10b" / "emergence_curves.json")
    if e10b:
        a = e10b.get("angle_vs_layer", {})
        k = e10b.get("kl_vs_layer", {})
        ok = bool(a) and bool(k) and all(v == v for v in a.values()) and all(v == v for v in k.values())
        add("Exp10b emergence curves", ok, {
            "layers_angle": a,
            "layers_kl": k,
        })
    else:
        add("Exp10b emergence curves", False, "missing")

    # Print compact table
    print("\nACCEPTANCE SUMMARY")
    print("------------------")
    for name, status, detail in rows:
        print(f"{status:4}  {name}  :: {detail}")


if __name__ == "__main__":
    main()

