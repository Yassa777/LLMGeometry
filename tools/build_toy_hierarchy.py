#!/usr/bin/env python3
"""
Write a tiny toy concept hierarchy JSON for smoke tests.

Usage:
  python tools/build_toy_hierarchy.py --out runs/exp01/concept_hierarchies.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Write a toy concept hierarchy JSON")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    data = [
        {
            "parent": {"synset_id": "p_animal", "name": "animal", "prompts": []},
            "children": [
                {"synset_id": "c_dog", "name": "dog", "prompts": []},
                {"synset_id": "c_cat", "name": "cat", "prompts": []},
            ],
            "parent_prompts": [
                "A photo of an animal.",
                "An animal in the wild.",
                "An animal in a zoo.",
                "A cute animal.",
                "An animal running.",
            ],
            "child_prompts": {
                "c_dog": [
                    "A photo of a dog.",
                    "A dog playing fetch.",
                    "A barking dog.",
                    "A puppy sleeping.",
                    "A golden retriever in a park.",
                ],
                "c_cat": [
                    "A photo of a cat.",
                    "A cat on a sofa.",
                    "A cat chasing a toy.",
                    "A kitten sleeping.",
                    "A black cat on a window.",
                ],
            },
        }
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Wrote toy hierarchy:", out_path)


if __name__ == "__main__":
    main()

