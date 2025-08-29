#!/usr/bin/env python3
"""
Generate GPU-tuned configs from base configs by overriding device/model and (optionally) n_prompts.

Writes to runs/gpu-configs/expXX.yaml.

Usage:
  python tools/make_gpu_configs.py --device cuda:0 --model distilgpt2 --n-prompts 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import yaml


EXPS = [f"exp{str(i).zfill(2)}.yaml" for i in range(1, 11)] + ["exp10b.yaml"]


def override(path_in: Path, path_out: Path, device: str, model: str, n_prompts: int | None) -> None:
    with open(path_in, "r") as f:
        cfg = yaml.safe_load(f)
    # Shallow overrides where present
    cfg.setdefault("run", {})
    cfg["run"]["device"] = device
    cfg.setdefault("model", {})
    cfg["model"]["name"] = model
    if n_prompts is not None:
        cfg.setdefault("data", {})
        cfg["data"]["n_prompts"] = int(n_prompts)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description="Make GPU configs")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--base", type=str, default="configs")
    ap.add_argument("--out", type=str, default="runs/gpu-configs")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in EXPS:
        src = base / name
        if not src.exists():
            continue
        # Only override for exps that use model/device
        override(src, out / name, args.device, args.model, args.n_prompts if name in ("exp06.yaml", "exp10.yaml") else None)

    print("Wrote GPU configs to:", out)


if __name__ == "__main__":
    main()
