#!/usr/bin/env bash
set -euo pipefail

# End-to-end setup + GPU run for LLMGeometry experiments with optional W&B logging.
#
# Usage:
#   bash scripts/setup_and_run_gpu.sh \
#     --model distilgpt2 \
#     --device cuda:0 \
#     [--hf-token $HUGGINGFACE_HUB_TOKEN] \
#     [--wandb-key $WANDB_API_KEY --project LLMGeometry --run-name exp_run] \
#     [--hier /path/to/hierarchy.json] \
#     [--max-pos 8 --max-neg 8 --n-prompts 64]

MODEL="EleutherAI/pythia-410m-deduped"
DEVICE="cuda:0"
HF_TOKEN=""
WANDB_KEY=""
WANDB_PROJECT=""
WANDB_RUN_NAME="llmgeom_all"
HIER_JSON=""
HIER_SOURCE="default" # default|wordnet
MAX_POS=8
MAX_NEG=8
N_PROMPTS=64
BUILD_POOLED=1
WN_PARENTS="animal.n.01,vehicle.n.01,profession.n.01,food.n.01,emotion.n.01,programming_language.n.01,geographical_area.n.01,art.n.01"
WN_CHILDREN=6
WN_PROMPTS=24
WN_MIN_ZIPF=3.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --hf-token) HF_TOKEN="$2"; shift 2;;
    --wandb-key) WANDB_KEY="$2"; shift 2;;
    --project) WANDB_PROJECT="$2"; shift 2;;
    --run-name) WANDB_RUN_NAME="$2"; shift 2;;
    --hier) HIER_JSON="$2"; shift 2;;
    --hier-source) HIER_SOURCE="$2"; shift 2;;
    --max-pos) MAX_POS="$2"; shift 2;;
    --max-neg) MAX_NEG="$2"; shift 2;;
    --n-prompts) N_PROMPTS="$2"; shift 2;;
    --build-pooled) BUILD_POOLED=1; shift 1;;
    --no-build-pooled) BUILD_POOLED=0; shift 1;;
    --wn-parents) WN_PARENTS="$2"; shift 2;;
    --wn-children) WN_CHILDREN="$2"; shift 2;;
    --wn-prompts) WN_PROMPTS="$2"; shift 2;;
    --wn-min-zipf) WN_MIN_ZIPF="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

echo "[1/8] Installing requirements..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt >/dev/null

if [[ -n "$HF_TOKEN" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  echo "Hugging Face token configured via env."
fi

if [[ -n "$WANDB_KEY" ]]; then
  export WANDB_API_KEY="$WANDB_KEY"
  echo "W&B API key configured via env."
fi

echo "[2/8] Preparing hierarchy JSON..."
if [[ -n "$HIER_JSON" ]]; then
  mkdir -p runs/exp01
  cp -f "$HIER_JSON" runs/exp01/concept_hierarchies.json
elif [[ "$HIER_SOURCE" == "wordnet" ]]; then
  echo "Building WordNet-backed hierarchy..."
  PYTHONPATH=. python tools/build_wordnet_hierarchy.py \
    --out runs/exp01/concept_hierarchies.json \
    --parents "$WN_PARENTS" \
    --children-per-parent "$WN_CHILDREN" \
    --prompts-per-concept "$WN_PROMPTS" \
    --min-zipf "$WN_MIN_ZIPF"
else
  PYTHONPATH=. python tools/build_default_hierarchy.py --out runs/exp01/concept_hierarchies.json
fi

echo "[3/8] Building hierarchical activations on $DEVICE..."
PYTHONPATH=. python tools/build_activations.py \
  --model "$MODEL" \
  --hier runs/exp01/concept_hierarchies.json \
  --out runs/exp01/activations.h5 \
  --device "$DEVICE" \
  --max-pos "$MAX_POS" \
  --max-neg "$MAX_NEG"

echo "[4/8] Generating GPU configs (device=$DEVICE, model=$MODEL, n_prompts=$N_PROMPTS)..."
PYTHONPATH=. python tools/make_gpu_configs.py --device "$DEVICE" --model "$MODEL" --n-prompts "$N_PROMPTS"

echo "[5/8] Running experiments 01–10..."
# Exp01 uses exp01.yaml; others read from default configs except those needing device/model overrides.
PYTHONPATH=. python experiments/exp01_angles.py --config runs/gpu-configs/exp01.yaml
echo "[5a] Ensuring pooled TV availability for Exp09..."
if [[ -f runs/exp01_pooled/teacher_vectors.json ]]; then
  echo "Using existing pooled TV at runs/exp01_pooled/teacher_vectors.json"
elif [[ "$BUILD_POOLED" == "1" ]]; then
  echo "Building pooled activations and pooled teacher vectors..."
  mkdir -p runs/exp01_pooled
  PYTHONPATH=. python tools/build_activations.py \
    --model "$MODEL" \
    --hier runs/exp01/concept_hierarchies.json \
    --out runs/exp01_pooled/activations.h5 \
    --device "$DEVICE" \
    --max-pos "$MAX_POS" \
    --max-neg "$MAX_NEG" \
    --granularity pooled
  # Create pooled config for Exp01 by overriding activations path and save_dir
  mkdir -p runs/gpu-configs
  cat > runs/gpu-configs/exp01_pooled.yaml << EOF
run:
  device: $DEVICE
model:
  name: $MODEL
geometry:
  shrinkage: 0.05
  lda_shrinkage: 0.10
concepts:
  file: runs/exp01/concept_hierarchies.json
data:
  activations: runs/exp01_pooled/activations.h5
logging:
  save_dir: runs/exp01_pooled
eval:
  angle_threshold_deg: 80
EOF
  PYTHONPATH=. python experiments/exp01_angles.py --config runs/gpu-configs/exp01_pooled.yaml
else
  echo "Pooled TV missing and --build-pooled not set; creating placeholder from last-token TV."
  mkdir -p runs/exp01_pooled
  cp -f runs/exp01/teacher_vectors.json runs/exp01_pooled/teacher_vectors.json
fi
PYTHONPATH=. python experiments/exp02_ratio_invariance.py --config configs/exp02.yaml
PYTHONPATH=. python experiments/exp03_euclid_vs_causal.py --config configs/exp03.yaml
PYTHONPATH=. python experiments/exp03_contrasts.py --config configs/exp03b.yaml
PYTHONPATH=. python experiments/exp04_boundary_normals.py --config configs/exp04.yaml
PYTHONPATH=. python experiments/exp05_interventions.py --config runs/gpu-configs/exp05.yaml
PYTHONPATH=. python experiments/exp05_estimators.py --config runs/gpu-configs/exp05b.yaml
PYTHONPATH=. python experiments/exp06_fisher_logit.py --config runs/gpu-configs/exp06.yaml
PYTHONPATH=. python experiments/exp07_whitening_ablation.py --config runs/gpu-configs/exp07.yaml
PYTHONPATH=. python experiments/exp08_dataset_variants.py --config configs/exp08.yaml
PYTHONPATH=. python experiments/exp09_token_granularity.py --config configs/exp09.yaml
PYTHONPATH=. python experiments/exp10_layer_variants.py --config runs/gpu-configs/exp10.yaml
PYTHONPATH=. python experiments/exp10_emergence_curves.py --config runs/gpu-configs/exp10b.yaml

echo "[6/8] Building figures..."
PYTHONPATH=. python tools/figures.py --base runs

if [[ -n "$WANDB_PROJECT" ]]; then
  echo "[7/8] Logging results to W&B project=$WANDB_PROJECT run=$WANDB_RUN_NAME..."
  PYTHONPATH=. python tools/log_results_wandb.py --base runs --project "$WANDB_PROJECT" --run-name "$WANDB_RUN_NAME"
else
  echo "[7/8] W&B project not set; skipping logging. Use --project and --wandb-key to enable."
fi

echo "[8/8] Done. Outputs under runs/, figures under runs/figures/."
