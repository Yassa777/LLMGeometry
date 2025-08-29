#!/usr/bin/env python3
"""
Run selected geometry-only experiments in sequence.

Examples:
  python experiments/run_all.py --exp01 --config01 configs/exp01.yaml
  python experiments/run_all.py --exp01 --exp02 --exp03 \
      --config01 configs/exp01.yaml --config02 configs/exp02.yaml --config03 configs/exp03.yaml

If no flags provided, runs --exp01 --exp02 --exp03 with default configs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run geometry experiments")
    ap.add_argument("--exp01", action="store_true", help="Run Exp01: angles")
    ap.add_argument("--exp02", action="store_true", help="Run Exp02: ratio invariance")
    ap.add_argument("--exp02b", action="store_true", help="Run Exp02b: ratio invariance (logits)")
    ap.add_argument("--exp03", action="store_true", help="Run Exp03: euclid vs causal")
    ap.add_argument("--exp03b", action="store_true", help="Run Exp03b: contrasts (LDA vs mean-diff)")
    ap.add_argument("--exp04", action="store_true", help="Run Exp04: boundary normals")
    ap.add_argument("--exp05", action="store_true", help="Run Exp05: interventions")
    ap.add_argument("--exp06", action="store_true", help="Run Exp06: fisher/logit")
    ap.add_argument("--exp07", action="store_true", help="Run Exp07: whitening ablation")
    ap.add_argument("--exp08", action="store_true", help="Run Exp08: dataset variants")
    ap.add_argument("--exp09", action="store_true", help="Run Exp09: token granularity")
    ap.add_argument("--exp10", action="store_true", help="Run Exp10: layer variants")
    ap.add_argument("--config01", type=str, default="configs/exp01.yaml")
    ap.add_argument("--config02", type=str, default="configs/exp02.yaml")
    ap.add_argument("--config02b", type=str, default="configs/exp02b.yaml")
    ap.add_argument("--config03", type=str, default="configs/exp03.yaml")
    ap.add_argument("--config03b", type=str, default="configs/exp03b.yaml")
    ap.add_argument("--config04", type=str, default="configs/exp04.yaml")
    ap.add_argument("--config05", type=str, default="configs/exp05.yaml")
    ap.add_argument("--config06", type=str, default="configs/exp06.yaml")
    ap.add_argument("--config07", type=str, default="configs/exp07.yaml")
    ap.add_argument("--config08", type=str, default="configs/exp08.yaml")
    ap.add_argument("--config09", type=str, default="configs/exp09.yaml")
    ap.add_argument("--config10", type=str, default="configs/exp10.yaml")
    args = ap.parse_args()

    # Default: run all
    do_all = not (args.exp01 or args.exp02 or args.exp03 or args.exp04 or args.exp05 or args.exp06 or args.exp07 or args.exp08 or args.exp09 or args.exp10)

    root = Path(__file__).parent
    py = sys.executable
    rc = 0

    if args.exp01 or do_all:
        rc = run([py, str(root / "exp01_angles.py"), "--config", args.config01])
        if rc != 0:
            sys.exit(rc)
    if args.exp02 or do_all:
        rc = run([py, str(root / "exp02_ratio_invariance.py"), "--config", args.config02])
        if rc != 0:
            sys.exit(rc)
    if args.exp03 or do_all:
        rc = run([py, str(root / "exp03_euclid_vs_causal.py"), "--config", args.config03])
        if rc != 0:
            sys.exit(rc)
    if args.exp02b or do_all:
        rc = run([py, str(root / "exp02b_ratio_invariance_logits.py"), "--config", args.config02b])
        if rc != 0:
            sys.exit(rc)
    if args.exp04 or do_all:
        rc = run([py, str(root / "exp04_boundary_normals.py"), "--config", args.config04])
        if rc != 0:
            sys.exit(rc)
    if args.exp03b or do_all:
        rc = run([py, str(root / "exp03_contrasts.py"), "--config", args.config03b])
        if rc != 0:
            sys.exit(rc)
    if args.exp05 or do_all:
        rc = run([py, str(root / "exp05_interventions.py"), "--config", args.config05])
        if rc != 0:
            sys.exit(rc)
    if args.exp06 or do_all:
        rc = run([py, str(root / "exp06_fisher_logit.py"), "--config", args.config06])
        if rc != 0:
            sys.exit(rc)
    if args.exp07 or do_all:
        rc = run([py, str(root / "exp07_whitening_ablation.py"), "--config", args.config07])
        if rc != 0:
            sys.exit(rc)
    if args.exp08 or do_all:
        rc = run([py, str(root / "exp08_dataset_variants.py"), "--config", args.config08])
        if rc != 0:
            sys.exit(rc)
    if args.exp09 or do_all:
        rc = run([py, str(root / "exp09_token_granularity.py"), "--config", args.config09])
        if rc != 0:
            sys.exit(rc)
    if args.exp10 or do_all:
        rc = run([py, str(root / "exp10_layer_variants.py"), "--config", args.config10])
        if rc != 0:
            sys.exit(rc)

    print("All selected experiments completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
