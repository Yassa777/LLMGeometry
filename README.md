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

## Status

This is a fresh scaffold; geometry + metrics land next.
