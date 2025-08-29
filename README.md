# LLMGeometry

A lean, geometry-only toolkit for final-layer analyses in language models:
whitening (ZCA), causal inner products, causal angles, ratio-invariance,
boundary-normal alignment, and LM interventions — without SAEs.

This repo implements the experiments outlined in LLMGeometry.pdf using
HF models and a simple Phase-1-style pipeline.

## Layout

- `llmgeometry/` — core library (geometry, metrics, activations, concepts,
  estimators, validation, interventions)
- `experiments/` — scripts for Exp01–Exp10
- `configs/` — base/component configs and per-experiment configs
- `tools/` — diagnostics and figure generation
- `tests/` — unit and smoke tests
- `scripts/` — install/run helpers

## Getting Started

- `pip install -r requirements.txt`
- Run Experiment 1 (angles + geometry validation):
  - `python experiments/exp01_angles.py --config configs/exp01.yaml`
- Build figures (after running experiments):
  - `python tools/figures.py --base runs`
    - Produces PNGs under `runs/figures/` for Exp01–Exp10, including Exp02–Exp06.

## End-to-End GPU Run

- One-shot setup + full run with optional W&B logging:
  - `bash scripts/setup_and_run_gpu.sh --model distilgpt2 --device cuda:0 \
     --hf-token $HUGGINGFACE_HUB_TOKEN \
     --wandb-key $WANDB_API_KEY --project LLMGeometry --run-name full_run`
  - Flags:
    - `--hier /path/to/concept_hierarchies.json` Use your real hierarchy (optional; a toy one is created if omitted for smoke use).
    - `--max-pos/--max-neg` to limit activations per concept (defaults 8)
    - `--n-prompts` to cap prompts for Exp06/Exp10 (default 64)
    - `--build-pooled` to build a pooled-token variant for Exp09 (otherwise uses placeholder if `runs/exp01_pooled/teacher_vectors.json` absent)
  - Produces:
    - `runs/exp01..exp10/*` JSON results, `runs/figures/*.png`, and (if configured) logs to W&B.

## Preparing Inputs

- Concept hierarchies: provide a JSON at `runs/exp01/concept_hierarchies.json` with fields
  `parent`, `children`, `parent_prompts`, `child_prompts` (see `llmgeometry/concepts.py`).
- Hierarchical activations (HDF5):
  - Build from the hierarchy JSON via:
    - `python tools/build_activations.py --model distilgpt2 --hier runs/exp01/concept_hierarchies.json --out runs/exp01/activations.h5 --device cpu`
  - The script populates per-concept positive/negative activations (heuristic negatives from siblings/others).

### Building Hierarchies

- Curated default (multi-domain):
  - `python tools/build_default_hierarchy.py --out runs/exp01/concept_hierarchies.json`
- WordNet-backed (recommended for scale):
  - `python tools/build_wordnet_hierarchy.py --out runs/exp01/concept_hierarchies.json \
       --children-per-parent 6 --prompts-per-concept 24 --min-zipf 3.0`
  - Customize parents via `--parents animal.n.01,vehicle.n.01,...`.
  - In the end-to-end script, use: `--hier-source wordnet` and optional
    `--wn-children/--wn-prompts/--wn-min-zipf/--wn-parents`.

## Experiments Overview

- Exp01: Teacher vectors & causal angles
  - Tests: parent vectors via LDA, child deltas, causal angle stats.
  - Metrics: `angle_stats.*` (median, fraction_above_threshold); `geometry_stats.*` (whitening invariant).
  - Outputs: `runs/exp01/teacher_vectors.json`, `runs/exp01/geometry.pt`.
  - Figures: `fig_angles_hist.png`.

- Exp02: Ratio-invariance (synthetic)
  - Tests: KL divergence of child magnitude distribution under parent-direction interventions.
  - Metrics: `summary.median_kl`, `summary.fraction_below_0_1`.
  - Outputs: `runs/exp02/ratio_invariance.json`.
  - Figures: `fig_ratio_invariance.png`, `fig_ratio_invariance_per_parent.png`, per-parent grid under `ratio_invariance_per_parent/`.

- Exp03: Euclidean vs Causal angles
  - Tests: comparison of angle medians and orthogonality fraction.
  - Metrics: `euclidean.median`, `causal.median`, `improvement.*`.
  - Outputs: `runs/exp03/euclid_vs_causal.json`.
  - Figures: `fig_euclid_vs_causal.png`.

- Exp04: Boundary-normal alignment
  - Tests: alignment between logistic boundary normals and teacher child deltas.
  - Metrics: `summary.median_of_medians`, `per_parent_median.*`.
  - Outputs: `runs/exp04/boundary_normals.json`.
  - Figures: `fig_boundary_normals.png`, `fig_boundary_normals_per_parent.png`.

- Exp05: Interventions
  - Tests: parent-vector interventions and effect size on logits.
  - Metrics: per-magnitude median mean|Δlogits| across parents.
  - Outputs: `runs/exp05/interventions.json` (includes per-prompt stats for scatter).
  - Figures: `fig_interventions.png`, `fig_interventions_scatter.png`.

- Exp06: Fisher/Logit diagonal approximations
  - Tests: diagonal approximations mapped to residual space via unembedding; angle stats vs baseline.
  - Metrics: `{baseline,fisher_diag,logit_var_diag}.{median,fraction_above_80}`.
  - Outputs: `runs/exp06/fisher_logit_summary.json`.
  - Figures: `fig_fisher_logit.png`, `fig_fisher_logit_delta.png`.

- Exp07: Whitening ablation
  - Tests: effect of shrinkage on whitening invariant and angle median.
  - Metrics: `results[shrinkage].{W_Sigma_Wt_minus_I_*, angle_median_deg}`.
  - Outputs: `runs/exp07/whitening_ablation.json`.
  - Figures: `fig_whitening_ablation.png`.

- Exp08: Dataset variants
  - Tests: angle medians across multiple teacher_vectors.json variants.
  - Metrics: `summary.median_of_medians`.
  - Outputs: `runs/exp08/dataset_variants.json`.
  - Figures: `fig_dataset_variants.png`.

- Exp09: Token granularity
  - Tests: last-token vs pooled features.
  - Metrics: `last_token.angle_median_deg`, `pooled.angle_median_deg`, `delta`.
  - Outputs: `runs/exp09/token_granularity.json`.
  - Figures: `fig_token_granularity.png`.

- Exp10: Layer variants
  - Tests: layer-wise covariances over prompts; angle medians per layer.
  - Metrics: `layers[layer_idx].angle_median_deg`.
  - Outputs: `runs/exp10/layer_variants.json`.
  - Figures: `fig_layer_variants.png`.

Pass/Fail Guidance
- These are research diagnostics; default thresholds vary by model/data. As quick sanity:
  - Whitening invariant MAE small (~1e-2–1e-1 range depending on shrinkage/model).
  - Causal > Euclid angle median in Exp03.
  - Ratio-invariance median KL reasonably low (< 0.2) in Exp02.
  - Boundary-normal medians comfortably above 60° in Exp04.
  - Interventions effect grows with magnitude in Exp05.
  - Diagonal approximations do not catastrophically degrade angles in Exp06.

## W&B Logging

- Post-run logging:
  - `python tools/log_results_wandb.py --base runs --project <PROJECT> --run-name <NAME>`
  - The end-to-end script (`scripts/setup_and_run_gpu.sh`) calls this automatically when `--project` and `--wandb-key` are provided.

## Status

This is a fresh scaffold; geometry + metrics land next.
