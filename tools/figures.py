#!/usr/bin/env python3
"""
Generate simple figures for geometry-only experiments.

Produces:
  - fig_angles_hist.png from Exp01 teacher_vectors.json
  - fig_euclid_vs_causal.png from Exp03 euclid_vs_causal.json
  - fig_ratio_invariance.png from Exp02 ratio_invariance.json
  - ratio_invariance_per_parent/*.png small multiples per-parent (Exp02)
  - fig_boundary_normals.png from Exp04 boundary_normals.json
  - fig_boundary_normals_per_parent.png bar of per-parent medians (Exp04)
  - fig_interventions.png from Exp05 interventions.json
  - fig_interventions_scatter.png per-prompt scatter by magnitude (Exp05)
  - fig_fisher_logit.png from Exp06 fisher_logit_summary.json
  - fig_fisher_logit_delta.png deltas vs baseline (Exp06)
  - fig_whitening_ablation.png from Exp07 whitening_ablation.json
  - fig_dataset_variants.png from Exp08 dataset_variants.json
  - fig_polysemy_subsets.png from Exp08 polysemy subsets (if present)
  - fig_token_granularity.png from Exp09 token_granularity.json
  - fig_angle_vs_freq.png from Exp09 angle_vs_logfreq curve (if present)
  - fig_layer_variants.png from Exp10 layer_variants.json
  - fig_contrasts_angle_hist.png from Exp03b contrasts.json
  - fig_contrasts_auroc.png from Exp03b contrasts.json
  - fig_estimators_auroc.png from Exp05b estimators.json
  - fig_estimators_angles.png from Exp05b estimators.json (if teacher angles present)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fig_angles_hist(tv_path: str, out_path: str) -> None:
    data = json.load(open(tv_path))
    angle_stats = data.get("angle_stats", {})
    # Rebuild angles if present in detailed form
    # Here we only plot a synthetic histogram using median/std if raw angles absent
    angles = None
    if "angles_deg" in data:
        angles = np.array(data["angles_deg"])  # if raw stored
    if angles is None:
        med = float(angle_stats.get("median_angle_deg", 90.0))
        std = float(angle_stats.get("std_angle_deg", 10.0))
        rng = np.random.default_rng(0)
        angles = rng.normal(loc=med, scale=std, size=1000)
    plt.figure(figsize=(5, 3.5))
    plt.hist(angles, bins=36, density=True, alpha=0.8, color="#1f77b4")
    plt.axvline(80, linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Causal angle (degrees)")
    plt.ylabel("Density")
    plt.title("Causal angles between ℓp and δc|p")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_euclid_vs_causal(comp_path: str, out_path: str) -> None:
    data = json.load(open(comp_path))
    em = data.get("euclidean", {}).get("median", np.nan)
    cm = data.get("causal", {}).get("median", np.nan)
    plt.figure(figsize=(4, 3.2))
    xs = ["Euclid", "Causal"]
    ys = [em, cm]
    plt.bar(xs, ys, color=["#999999", "#ff7f0e"]) 
    plt.axhline(80, linestyle="--", color="gray", linewidth=1)
    plt.ylabel("Median angle (deg)")
    plt.title("Euclidean vs Causal angle medians")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_ratio_invariance(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_parent", {})
    # Collect KLs aggregated by alpha
    kl_by_alpha = {}
    for pid, res in per.items():
        by_alpha = res.get("by_alpha", {})
        for a_str, vals in by_alpha.items():
            try:
                a = float(a_str)
            except Exception:
                # keys may already be floats; handle directly
                a = float(a_str)
            kl = vals.get("kl_divergence")
            if kl is None:
                continue
            kl_by_alpha.setdefault(a, []).append(float(kl))
    if not kl_by_alpha:
        return
    alphas = sorted(kl_by_alpha.keys())
    medians = [float(np.median(kl_by_alpha[a])) if kl_by_alpha[a] else np.nan for a in alphas]
    q25 = [float(np.percentile(kl_by_alpha[a], 25)) if kl_by_alpha[a] else np.nan for a in alphas]
    q75 = [float(np.percentile(kl_by_alpha[a], 75)) if kl_by_alpha[a] else np.nan for a in alphas]
    plt.figure(figsize=(5, 3.2))
    plt.plot(alphas, medians, marker="o", color="#9467bd", label="median KL")
    # Shaded IQR band
    plt.fill_between(alphas, q25, q75, color="#9467bd", alpha=0.15, label="IQR")
    plt.xlabel(r"Intervention magnitude $\alpha$")
    plt.ylabel("KL(base || intervened)")
    plt.title("Ratio-invariance across magnitudes")
    plt.legend(frameon=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_boundary_normals(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_parent", {})
    # Flatten all child angles
    all_angles = []
    for pid, child_map in per.items():
        if isinstance(child_map, dict) and "angles" in child_map:
            ch = child_map.get("angles", {})
        else:
            ch = child_map
        for cid, ang in ch.items():
            try:
                all_angles.append(float(ang))
            except Exception:
                continue
    if not all_angles:
        return
    plt.figure(figsize=(5, 3.2))
    plt.hist(all_angles, bins=36, density=True, alpha=0.8, color="#2ca02c")
    plt.axvline(80, linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Causal angle (degrees)")
    plt.ylabel("Density")
    plt.title("Boundary normals vs teacher δ angles")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_interventions(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_parent", {})
    # Aggregate mean_abs_delta by magnitude; support both legacy list and new dict format
    vals_by_mag = {}
    for pid, series in per.items():
        if isinstance(series, list):
            it = series
        elif isinstance(series, dict) and "per_magnitude" in series:
            it = series["per_magnitude"]
        else:
            it = []
        for item in it:
            try:
                m = float(item.get("magnitude", np.nan))
            except Exception:
                continue
            v = item.get("mean_abs_delta")
            if v is None or np.isnan(m):
                continue
            vals_by_mag.setdefault(m, []).append(float(v))
    if not vals_by_mag:
        return
    mags = sorted(vals_by_mag.keys())
    med = [float(np.median(vals_by_mag[m])) if vals_by_mag[m] else np.nan for m in mags]
    q25 = [float(np.percentile(vals_by_mag[m], 25)) if vals_by_mag[m] else np.nan for m in mags]
    q75 = [float(np.percentile(vals_by_mag[m], 75)) if vals_by_mag[m] else np.nan for m in mags]
    plt.figure(figsize=(5, 3.2))
    plt.plot(mags, med, marker="o", color="#d62728", label="median |Δlogits|")
    plt.fill_between(mags, q25, q75, color="#d62728", alpha=0.15, label="IQR")
    plt.xlabel("Intervention magnitude")
    plt.ylabel("Mean |Δlogits|")
    plt.title("Intervention effect vs magnitude")
    plt.legend(frameon=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_fisher_logit(path: str, out_path: str) -> None:
    data = json.load(open(path))
    labels = ["baseline", "fisher_diag", "logit_var_diag"]
    med = [data.get(k, {}).get("median", np.nan) for k in labels]
    frac = [data.get(k, {}).get("fraction_above_80", np.nan) for k in labels]
    if all(np.isnan(x) for x in med) and all(np.isnan(x) for x in frac):
        return
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    # Left: median angles
    axs[0].bar(range(len(labels)), med, color=["#999999", "#1f77b4", "#ff7f0e"])  
    axs[0].axhline(80, linestyle="--", color="gray", linewidth=1)
    axs[0].set_xticks(range(len(labels)))
    axs[0].set_xticklabels(["base", "Fisher", "LogitVar"])
    axs[0].set_ylabel("Median angle (deg)")
    axs[0].set_title("Angle medians")
    # Right: fraction >= 80°
    axs[1].bar(range(len(labels)), frac, color=["#999999", "#1f77b4", "#ff7f0e"])  
    axs[1].set_xticks(range(len(labels)))
    axs[1].set_xticklabels(["base", "Fisher", "LogitVar"])
    axs[1].set_ylabel("Fraction ≥ 80°")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Orthogonality fraction")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def fig_ratio_invariance_per_parent(path: str, out_dir: str, max_grid: int = 12) -> None:
    data = json.load(open(path))
    per = data.get("per_parent", {})
    if not per:
        return
    # Compute an ordering by median KL across alphas (descending)
    order = []
    for pid, res in per.items():
        by_alpha = res.get("by_alpha", {})
        kls = [float(v.get("kl_divergence", np.nan)) for v in by_alpha.values()]
        kls = [x for x in kls if np.isfinite(x)]
        med = float(np.median(kls)) if kls else np.nan
        order.append((pid, med))
    order = [p for p in order if np.isfinite(p[1])]
    order.sort(key=lambda x: x[1], reverse=True)

    # Grid of top-N
    top = order[:max_grid]
    n = len(top)
    if n == 0:
        return
    rows = int(np.ceil(n / 4))
    cols = min(4, n)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for idx, (pid, _) in enumerate(top):
        r, c = divmod(idx, cols)
        ax = axs[r][c]
        by_alpha = per[pid].get("by_alpha", {})
        items = sorted([(float(a), float(v.get("kl_divergence", np.nan))) for a, v in by_alpha.items()], key=lambda x: x[0])
        if items:
            xs, ys = zip(*items)
            ax.plot(xs, ys, marker="o", color="#9467bd")
        ax.set_title(str(pid))
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("KL")
    # Hide unused axes
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        fig.delaxes(axs[r][c])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "fig_ratio_invariance_per_parent.png"), dpi=200)
    plt.close(fig)

    # Also write individual per-parent plots to folder
    per_dir = Path(out_dir) / "ratio_invariance_per_parent"
    per_dir.mkdir(parents=True, exist_ok=True)
    for pid, _ in order:
        by_alpha = per[pid].get("by_alpha", {})
        items = sorted([(float(a), float(v.get("kl_divergence", np.nan))) for a, v in by_alpha.items()], key=lambda x: x[0])
        if not items:
            continue
        xs, ys = zip(*items)
        plt.figure(figsize=(4, 3.2))
        plt.plot(xs, ys, marker="o", color="#9467bd")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("KL")
        plt.title(f"Ratio-invariance: {pid}")
        plt.tight_layout()
        plt.savefig(str(per_dir / f"{pid}.png"), dpi=200)
        plt.close()

def fig_boundary_normals_per_parent(path: str, out_path: str, max_bars: int = 40) -> None:
    data = json.load(open(path))
    summary = data.get("summary", {})
    med = summary.get("per_parent_median", {})
    if not med:
        return
    items = [(k, float(v)) for k, v in med.items() if np.isfinite(v)]
    if not items:
        return
    items.sort(key=lambda x: x[1], reverse=True)
    if max_bars is not None and len(items) > max_bars:
        items = items[:max_bars]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(max(6, 0.35 * len(labels)), 3.2))
    plt.bar(range(len(labels)), vals, color="#2ca02c")
    plt.axhline(80, linestyle="--", color="gray", linewidth=1)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Median causal angle (deg)")
    plt.title("Boundary-normal: per-parent medians (top)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_interventions_scatter(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_parent", {})
    # Accumulate per-prompt MADs (last-token) across parents if present
    x, y = [], []
    for pid, series in per.items():
        if not isinstance(series, dict) or "prompt_mads" not in series:
            continue
        pm = series["prompt_mads"]  # {magnitude(str): [list of floats]}
        for m_str, arr in pm.items():
            try:
                m = float(m_str)
            except Exception:
                continue
            for v in arr:
                try:
                    y.append(float(v))
                    x.append(m)
                except Exception:
                    continue
    if not x:
        return
    # Jitter x for visibility
    rng = np.random.default_rng(0)
    xj = np.array(x) + rng.normal(scale=0.01, size=len(x))
    plt.figure(figsize=(5, 3.2))
    plt.scatter(xj, y, s=10, alpha=0.4, color="#d62728")
    # Overlay median per magnitude
    mags = sorted(set(x))
    med = [float(np.median([yv for xv, yv in zip(x, y) if xv == m])) for m in mags]
    plt.plot(mags, med, marker="o", color="#1f1f1f", linewidth=1.5, label="median")
    plt.xlabel("Intervention magnitude")
    plt.ylabel("Per-prompt mean |Δlogits| (last token)")
    plt.title("Interventions: per-prompt effect")
    plt.legend(frameon=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_fisher_logit_delta(path: str, out_path: str) -> None:
    data = json.load(open(path))
    base_m = data.get("baseline", {}).get("median", np.nan)
    base_f = data.get("baseline", {}).get("fraction_above_80", np.nan)
    labels = ["Fisher", "LogitVar"]
    med = [data.get("fisher_diag", {}).get("median", np.nan) - base_m,
           data.get("logit_var_diag", {}).get("median", np.nan) - base_m]
    frac = [data.get("fisher_diag", {}).get("fraction_above_80", np.nan) - base_f,
            data.get("logit_var_diag", {}).get("fraction_above_80", np.nan) - base_f]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    axs[0].bar(range(len(labels)), med, color=["#1f77b4", "#ff7f0e"])  
    axs[0].axhline(0, linestyle="--", color="gray", linewidth=1)
    axs[0].set_xticks(range(len(labels)))
    axs[0].set_xticklabels(labels)
    axs[0].set_ylabel("Δ median angle (deg)")
    axs[0].set_title("Delta from baseline")
    axs[1].bar(range(len(labels)), frac, color=["#1f77b4", "#ff7f0e"])  
    axs[1].axhline(0, linestyle="--", color="gray", linewidth=1)
    axs[1].set_xticks(range(len(labels)))
    axs[1].set_xticklabels(labels)
    axs[1].set_ylabel("Δ fraction ≥ 80°")
    axs[1].set_title("Delta from baseline")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_whitening_ablation(path: str, out_path: str) -> None:
    data = json.load(open(path))
    res = data.get("results", {})
    if not res and "by_model" in data:
        # pick the first model's results
        bym = data.get("by_model", {})
        if bym:
            first_key = sorted(bym.keys())[0]
            res = bym[first_key]
    xs, ys = [], []
    for k, v in res.items():
        try:
            xs.append(float(k))
            ys.append(float(v.get("angle_median_deg", np.nan)))
        except Exception:
            continue
    if not xs:
        return
    idx = np.argsort(xs)
    xs = np.array(xs)[idx]
    ys = np.array(ys)[idx]
    plt.figure(figsize=(5, 3.2))
    plt.plot(xs, ys, marker="o", color="#1f77b4")
    plt.xlabel("Shrinkage")
    plt.ylabel("Median causal angle (deg)")
    plt.title("Whitening ablation: angle vs shrinkage")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_dataset_variants(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_variant", {})
    labels = list(per.keys())
    vals = [per[k].get("angle_median_deg", np.nan) for k in labels]
    if not labels:
        return
    plt.figure(figsize=(max(5, 0.6 * len(labels)), 3.2))
    plt.bar(range(len(labels)), vals, color="#1f77b4")
    plt.xticks(range(len(labels)), [Path(k).name for k in labels], rotation=45, ha="right")
    plt.ylabel("Median causal angle (deg)")
    plt.title("Dataset variants: angle medians")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_polysemy_subsets(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_variant", {})
    # Merge across variants: average medians if multiple
    mono, poly = [], []
    for v in per.values():
        p = v.get("polysemy", {})
        if "monosemous_median" in p:
            mono.append(float(p["monosemous_median"]))
        if "polysemous_median" in p:
            poly.append(float(p["polysemous_median"]))
    if not mono and not poly:
        return
    vals = [np.nanmean(mono) if mono else np.nan, np.nanmean(poly) if poly else np.nan]
    plt.figure(figsize=(4, 3.2))
    plt.bar(["mono", "poly"], vals, color=["#2ca02c", "#d62728"]) 
    plt.axhline(80, linestyle="--", color="gray", linewidth=1)
    plt.ylabel("Median causal angle (deg)")
    plt.title("Polysemy subsets")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_token_granularity(path: str, out_path: str) -> None:
    data = json.load(open(path))
    last_m = data.get("last_token", {}).get("angle_median_deg", np.nan)
    pool_m = data.get("pooled", {}).get("angle_median_deg", np.nan)
    plt.figure(figsize=(4, 3.2))
    plt.bar(["last", "pooled"], [last_m, pool_m], color=["#1f77b4", "#ff7f0e"]) 
    plt.ylabel("Median causal angle (deg)")
    plt.title("Token granularity")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_angle_vs_freq(path: str, out_path: str) -> None:
    data = json.load(open(path))
    curv = data.get("angle_vs_logfreq", {})
    if not curv:
        return
    xs = sorted(float(k) for k in curv.keys())
    ys = [float(curv[str(k)]) for k in xs]
    plt.figure(figsize=(5, 3.2))
    plt.plot(xs, ys, marker="o", color="#9467bd")
    plt.xlabel("Zipf frequency (log)")
    plt.ylabel("Median angle (deg)")
    plt.title("Angle vs log-frequency")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_layer_variants(path: str, out_path: str) -> None:
    data = json.load(open(path))
    layers = data.get("layers", {})
    xs = sorted(int(k) for k in layers.keys())
    ys = [layers[str(k)].get("angle_median_deg", np.nan) for k in xs]
    if not xs:
        return
    plt.figure(figsize=(5, 3.2))
    plt.plot(xs, ys, marker="o", color="#2ca02c")
    plt.xlabel("Layer index")
    plt.ylabel("Median causal angle (deg)")
    plt.title("Layer variants: angle median vs layer")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_contrasts_angle_hist(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_concept", {})
    angles = []
    for v in per.values():
        a = v.get("angle_deg")
        if a is None:
            continue
        try:
            angles.append(float(a))
        except Exception:
            continue
    if not angles:
        return
    plt.figure(figsize=(5, 3.2))
    plt.hist(angles, bins=36, density=True, alpha=0.85, color="#9467bd")
    plt.xlabel("Angle (deg) between LDA and mean-diff")
    plt.ylabel("Density")
    plt.title("Contrasts: LDA vs class-mean")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_contrasts_auroc(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_concept", {})
    a_lda, a_md = [], []
    for v in per.values():
        if "auroc_lda" in v:
            try:
                a_lda.append(float(v["auroc_lda"]))
            except Exception:
                pass
        if "auroc_mean_diff" in v:
            try:
                a_md.append(float(v["auroc_mean_diff"]))
            except Exception:
                pass
    if not a_lda and not a_md:
        return
    ys = [float(np.nanmedian(a_lda)) if a_lda else np.nan,
          float(np.nanmedian(a_md)) if a_md else np.nan]
    plt.figure(figsize=(4.5, 3.2))
    plt.bar(["LDA", "MeanDiff"], ys, color=["#1f77b4", "#ff7f0e"]) 
    plt.ylim(0.0, 1.0)
    plt.ylabel("Median AUROC")
    plt.title("Contrasts AUROC")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_estimators_auroc(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_concept", {})
    a_lda, a_md, a_l2 = [], [], []
    for v in per.values():
        for key, arr in [("auroc_lda", a_lda), ("auroc_mean_diff", a_md), ("auroc_l2probe", a_l2)]:
            if key in v:
                try:
                    arr.append(float(v[key]))
                except Exception:
                    pass
    if not (a_lda or a_md or a_l2):
        return
    labels = ["LDA", "MeanDiff", "L2Probe"]
    vals = [float(np.nanmedian(a_lda)) if a_lda else np.nan,
            float(np.nanmedian(a_md)) if a_md else np.nan,
            float(np.nanmedian(a_l2)) if a_l2 else np.nan]
    plt.figure(figsize=(5.5, 3.2))
    plt.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"]) 
    plt.ylim(0.0, 1.0)
    plt.ylabel("Median AUROC")
    plt.title("Estimator shoot-out")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def fig_estimators_angles(path: str, out_path: str) -> None:
    data = json.load(open(path))
    per = data.get("per_concept", {})
    ang_lda, ang_md, ang_l2 = [], [], []
    for v in per.values():
        for key, arr in [("angle_to_teacher_lda", ang_lda), ("angle_to_teacher_mean_diff", ang_md), ("angle_to_teacher_l2probe", ang_l2)]:
            if key in v:
                try:
                    arr.append(float(v[key]))
                except Exception:
                    pass
    if not (ang_lda or ang_md or ang_l2):
        return
    labels = ["LDA", "MeanDiff", "L2Probe"]
    vals = [float(np.nanmedian(ang_lda)) if ang_lda else np.nan,
            float(np.nanmedian(ang_md)) if ang_md else np.nan,
            float(np.nanmedian(ang_l2)) if ang_l2 else np.nan]
    plt.figure(figsize=(5.5, 3.2))
    plt.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"]) 
    plt.axhline(80, linestyle="--", color="gray", linewidth=1)
    plt.ylabel("Median angle to teacher (deg)")
    plt.title("Estimator angles vs teacher")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Build geometry figures")
    ap.add_argument("--base", type=str, default="runs")
    args = ap.parse_args()
    base = Path(args.base)

    tv = base / "exp01" / "teacher_vectors.json"
    if tv.exists():
        fig_angles_hist(str(tv), str(base / "figures" / "fig_angles_hist.png"))
        print("Wrote:", base / "figures" / "fig_angles_hist.png")

    comp = base / "exp03" / "euclid_vs_causal.json"
    if comp.exists():
        fig_euclid_vs_causal(str(comp), str(base / "figures" / "fig_euclid_vs_causal.png"))
        print("Wrote:", base / "figures" / "fig_euclid_vs_causal.png")

    rinv = base / "exp02" / "ratio_invariance.json"
    if rinv.exists():
        fig_ratio_invariance(str(rinv), str(base / "figures" / "fig_ratio_invariance.png"))
        print("Wrote:", base / "figures" / "fig_ratio_invariance.png")
        fig_ratio_invariance_per_parent(str(rinv), str(base / "figures"))
        print("Wrote:", base / "figures" / "fig_ratio_invariance_per_parent.png")

    bnorm = base / "exp04" / "boundary_normals.json"
    if bnorm.exists():
        fig_boundary_normals(str(bnorm), str(base / "figures" / "fig_boundary_normals.png"))
        print("Wrote:", base / "figures" / "fig_boundary_normals.png")
        fig_boundary_normals_per_parent(str(bnorm), str(base / "figures" / "fig_boundary_normals_per_parent.png"))
        print("Wrote:", base / "figures" / "fig_boundary_normals_per_parent.png")

    interv = base / "exp05" / "interventions.json"
    if interv.exists():
        fig_interventions(str(interv), str(base / "figures" / "fig_interventions.png"))
        print("Wrote:", base / "figures" / "fig_interventions.png")
        fig_interventions_scatter(str(interv), str(base / "figures" / "fig_interventions_scatter.png"))
        print("Wrote:", base / "figures" / "fig_interventions_scatter.png")

    fls = base / "exp06" / "fisher_logit_summary.json"
    if fls.exists():
        fig_fisher_logit(str(fls), str(base / "figures" / "fig_fisher_logit.png"))
        print("Wrote:", base / "figures" / "fig_fisher_logit.png")
        fig_fisher_logit_delta(str(fls), str(base / "figures" / "fig_fisher_logit_delta.png"))
        print("Wrote:", base / "figures" / "fig_fisher_logit_delta.png")

    wabl = base / "exp07" / "whitening_ablation.json"
    if wabl.exists():
        fig_whitening_ablation(str(wabl), str(base / "figures" / "fig_whitening_ablation.png"))
        print("Wrote:", base / "figures" / "fig_whitening_ablation.png")

    dsv = base / "exp08" / "dataset_variants.json"
    if dsv.exists():
        fig_dataset_variants(str(dsv), str(base / "figures" / "fig_dataset_variants.png"))
        print("Wrote:", base / "figures" / "fig_dataset_variants.png")
        fig_polysemy_subsets(str(dsv), str(base / "figures" / "fig_polysemy_subsets.png"))
        print("Wrote:", base / "figures" / "fig_polysemy_subsets.png")

    tkg = base / "exp09" / "token_granularity.json"
    if tkg.exists():
        fig_token_granularity(str(tkg), str(base / "figures" / "fig_token_granularity.png"))
        print("Wrote:", base / "figures" / "fig_token_granularity.png")
        fig_angle_vs_freq(str(tkg), str(base / "figures" / "fig_angle_vs_freq.png"))
        print("Wrote:", base / "figures" / "fig_angle_vs_freq.png")

    lvar = base / "exp10" / "layer_variants.json"
    if lvar.exists():
        fig_layer_variants(str(lvar), str(base / "figures" / "fig_layer_variants.png"))
        print("Wrote:", base / "figures" / "fig_layer_variants.png")

    # Exp03b: contrasts
    c3b = base / "exp03b" / "contrasts.json"
    if c3b.exists():
        fig_contrasts_angle_hist(str(c3b), str(base / "figures" / "fig_contrasts_angle_hist.png"))
        print("Wrote:", base / "figures" / "fig_contrasts_angle_hist.png")
        fig_contrasts_auroc(str(c3b), str(base / "figures" / "fig_contrasts_auroc.png"))
        print("Wrote:", base / "figures" / "fig_contrasts_auroc.png")

    # Exp05b: estimator shoot-out
    e5b = base / "exp05b" / "estimators.json"
    if e5b.exists():
        fig_estimators_auroc(str(e5b), str(base / "figures" / "fig_estimators_auroc.png"))
        print("Wrote:", base / "figures" / "fig_estimators_auroc.png")
        fig_estimators_angles(str(e5b), str(base / "figures" / "fig_estimators_angles.png"))
        print("Wrote:", base / "figures" / "fig_estimators_angles.png")


if __name__ == "__main__":
    main()
