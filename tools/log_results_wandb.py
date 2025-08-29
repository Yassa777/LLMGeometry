#!/usr/bin/env python3
"""
Log experiment outputs and figures to Weights & Biases.

Usage:
  python tools/log_results_wandb.py --base runs --project LLMGeometry --run-name exp_all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import wandb


def read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.load(open(p))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description="Log LLMGeometry results to W&B")
    ap.add_argument("--base", type=str, default="runs")
    ap.add_argument("--project", type=str, required=True)
    ap.add_argument("--run-name", type=str, default="llmgeom_all")
    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--tags", type=str, nargs="*", default=[])
    args = ap.parse_args()

    base = Path(args.base)
    run = wandb.init(project=args.project, name=args.run_name, notes=args.notes, tags=args.tags)

    # Exp01
    tv = read_json(base / "exp01" / "teacher_vectors.json")
    if tv:
        ang = tv.get("angle_stats", {})
        geom = tv.get("geometry_stats", {})
        wandb.log({
            "exp01/angle_median_deg": ang.get("median_angle_deg"),
            "exp01/fraction_above_80": ang.get("fraction_above_threshold"),
            "exp01/whiten_offdiag_rms": geom.get("whiten_offdiag_rms"),
            "exp01/whiten_fro_error": geom.get("whiten_fro_error"),
        })
        wandb.save(str(base / "exp01" / "teacher_vectors.json"))

    # Exp02
    ri = read_json(base / "exp02" / "ratio_invariance.json")
    if ri:
        s = ri.get("summary", {}) or ri.get("aggregate", {})
        wandb.log({
            "exp02/median_kl": s.get("median_kl"),
            "exp02/fraction_below_0.1": s.get("fraction_below_0_1"),
        })
        wandb.save(str(base / "exp02" / "ratio_invariance.json"))

    # Exp03
    e3 = read_json(base / "exp03" / "euclid_vs_causal.json")
    if e3:
        wandb.log({
            "exp03/euclid_median": e3.get("euclidean", {}).get("median"),
            "exp03/causal_median": e3.get("causal", {}).get("median"),
            "exp03/median_improvement": e3.get("improvement", {}).get("median_improvement"),
        })
        wandb.save(str(base / "exp03" / "euclid_vs_causal.json"))

    # Exp02b (logits ratio invariance)
    ri2 = read_json(base / "exp02b" / "ratio_invariance_logits.json")
    if ri2:
        s = ri2.get("summary", {})
        for k, v in s.items():
            if isinstance(v, dict):
                wandb.log({
                    f"exp02b/{k}_median_kl": v.get("median_kl"),
                    f"exp02b/{k}_median_parent_ddelta": v.get("median_parent_ddelta"),
                    f"exp02b/{k}_median_child_ddelta_abs": v.get("median_child_ddelta_abs"),
                })
        wandb.save(str(base / "exp02b" / "ratio_invariance_logits.json"))

    # Exp04
    b4 = read_json(base / "exp04" / "boundary_normals.json")
    if b4:
        s = b4.get("summary", {})
        wandb.log({
            "exp04/n_parents": s.get("n_parents"),
            "exp04/median_of_medians": s.get("median_of_medians"),
        })
        wandb.save(str(base / "exp04" / "boundary_normals.json"))

    # Exp05
    i5 = read_json(base / "exp05" / "interventions.json")
    if i5:
        per = i5.get("per_parent", {})
        # Compute median effect at each magnitude across parents
        agg: Dict[float, list] = {}
        for v in per.values():
            series = v.get("per_magnitude", v if isinstance(v, list) else [])
            for item in series:
                try:
                    m = float(item.get("magnitude"))
                    val = float(item.get("mean_abs_delta"))
                except Exception:
                    continue
                agg.setdefault(m, []).append(val)
        for m, vals in agg.items():
            if vals:
                wandb.log({f"exp05/mean_abs_delta_m{m}": sum(vals) / len(vals)})
        wandb.save(str(base / "exp05" / "interventions.json"))

    # Exp06
    f6 = read_json(base / "exp06" / "fisher_logit_summary.json")
    if f6:
        for k in ("baseline", "fisher_diag", "logit_var_diag"):
            v = f6.get(k, {})
            wandb.log({
                f"exp06/{k}_median": v.get("median"),
                f"exp06/{k}_frac80": v.get("fraction_above_80"),
            })
        wandb.save(str(base / "exp06" / "fisher_logit_summary.json"))

    # Exp03b contrasts
    c3b = read_json(base / "exp03b" / "contrasts.json")
    if c3b:
        s = c3b.get("summary", {})
        wandb.log({
            "exp03b/median_angle_deg": s.get("median_angle_deg"),
            "exp03b/median_auroc_lda": s.get("median_auroc_lda"),
            "exp03b/median_auroc_mean_diff": s.get("median_auroc_mean_diff"),
        })
        wandb.save(str(base / "exp03b" / "contrasts.json"))

    # Exp05b estimators
    e5b = read_json(base / "exp05b" / "estimators.json")
    if e5b:
        s = e5b.get("summary", {})
        wandb.log({
            "exp05b/median_auroc_lda": s.get("median_auroc_lda"),
            "exp05b/median_auroc_mean_diff": s.get("median_auroc_mean_diff"),
            "exp05b/median_auroc_l2probe": s.get("median_auroc_l2probe"),
        })
        wandb.save(str(base / "exp05b" / "estimators.json"))

    # Exp07
    w7 = read_json(base / "exp07" / "whitening_ablation.json")
    if w7:
        # Log best (max) angle median over shrinkages
        res = w7.get("results", {})
        vals = [v.get("angle_median_deg") for v in res.values() if isinstance(v, dict)]
        if vals:
            wandb.log({"exp07/angle_median_best": max(vals)})
        wandb.save(str(base / "exp07" / "whitening_ablation.json"))

    # Exp08
    d8 = read_json(base / "exp08" / "dataset_variants.json")
    if d8:
        s = d8.get("summary", {})
        wandb.log({"exp08/median_of_medians": s.get("median_of_medians")})
        wandb.save(str(base / "exp08" / "dataset_variants.json"))

    # Exp09
    t9 = read_json(base / "exp09" / "token_granularity.json")
    if t9:
        wandb.log({
            "exp09/last_token_median": t9.get("last_token", {}).get("angle_median_deg"),
            "exp09/pooled_median": t9.get("pooled", {}).get("angle_median_deg"),
            "exp09/delta": t9.get("delta"),
        })
        wandb.save(str(base / "exp09" / "token_granularity.json"))

    # Exp10
    l10 = read_json(base / "exp10" / "layer_variants.json")
    if l10:
        layers = l10.get("layers", {})
        for k, v in layers.items():
            wandb.log({f"exp10/layer_{k}_median": v.get("angle_median_deg")})
        wandb.save(str(base / "exp10" / "layer_variants.json"))

    # Exp10b emergence curves
    em = read_json(base / "exp10b" / "emergence_curves.json")
    if em:
        a = em.get("angle_vs_layer", {})
        aq25 = em.get("angle_vs_layer_q25", {})
        aq75 = em.get("angle_vs_layer_q75", {})
        k = em.get("kl_vs_layer", {})
        kq25 = em.get("kl_vs_layer_q25", {})
        kq75 = em.get("kl_vs_layer_q75", {})
        layers = sorted(set([int(x) for x in list(a.keys()) + list(k.keys())]))
        for layer in layers:
            wandb.log({
                f"exp10b/angle/median/L{layer}": a.get(str(layer)),
                f"exp10b/angle/q25/L{layer}": aq25.get(str(layer)),
                f"exp10b/angle/q75/L{layer}": aq75.get(str(layer)),
                f"exp10b/kl/median/L{layer}": k.get(str(layer)),
                f"exp10b/kl/q25/L{layer}": kq25.get(str(layer)),
                f"exp10b/kl/q75/L{layer}": kq75.get(str(layer)),
            })
        # Compact table with all per-layer stats
        n_a = em.get("n_angle_per_layer", {})
        n_k = em.get("n_kl_per_layer", {})
        table = wandb.Table(columns=[
            "layer", "angle_median", "angle_q25", "angle_q75",
            "kl_median", "kl_q25", "kl_q75", "n_angle", "n_kl"
        ])
        for layer in layers:
            row = [
                int(layer),
                a.get(str(layer)), aq25.get(str(layer)), aq75.get(str(layer)),
                k.get(str(layer)), kq25.get(str(layer)), kq75.get(str(layer)),
                n_a.get(str(layer)), n_k.get(str(layer)),
            ]
            table.add_data(*row)
        wandb.log({"exp10b/emergence_table": table})
        wandb.save(str(base / "exp10b" / "emergence_curves.json"))

    # Figures
    figs_dir = base / "figures"
    if figs_dir.exists():
        for p in figs_dir.glob("*.png"):
            try:
                wandb.log({f"fig/{p.stem}": wandb.Image(str(p))})
            except Exception:
                continue

    run.finish()


if __name__ == "__main__":
    main()
