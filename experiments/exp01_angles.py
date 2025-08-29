#!/usr/bin/env python3
"""
Exp01: Teacher vectors & causal angles.

Loads HF model to compute geometry (W, Sigma) from unembedding, loads
hierarchical activations (HDF5) and hierarchies (JSON), estimates parent
vectors and child deltas via LDA, and reports causal angle statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from llmgeometry import (
    CausalGeometry,
    ActivationCapture,
    ActivationConfig,
    Concept, ConceptHierarchy, load_concept_hierarchies, save_concept_hierarchies,
    LDAEstimator, ConceptVectorEstimator,
    hierarchical_orthogonality,
)


def load_unembedding(model_name: str, device: str = "cpu") -> torch.Tensor:
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device})
    if hasattr(mdl, "lm_head") and isinstance(mdl.lm_head, torch.nn.Module):
        U = mdl.lm_head.weight.detach().to("cpu")
    else:
        U = mdl.get_output_embeddings().weight.detach().to("cpu")
    del mdl
    return U


def main():
    ap = argparse.ArgumentParser(description="Exp01: Teacher vectors & causal angles")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg.get("logging", {}).get("save_dir", "runs/exp01"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load unembedding and build geometry
    U = load_unembedding(cfg["model"]["name"], device=str(cfg["run"]["device"]))
    geom = CausalGeometry(U, shrinkage=float(cfg.get("geometry", {}).get("shrinkage", 0.05)))
    torch.save({"Sigma": geom.Sigma.cpu(), "W": geom.W.cpu()}, out_dir / "geometry.pt")

    # Load hierarchies and activations
    hier_file = cfg["concepts"]["file"]
    acts_file = cfg["data"]["activations"]
    hierarchies = load_concept_hierarchies(hier_file)
    acts = ActivationCapture.load_hierarchical_activations(acts_file)

    # Split parent and child activations
    parent_acts = {}
    child_acts = {}
    parent_ids = {h.parent.synset_id for h in hierarchies}
    for cid, d in acts.items():
        if cid in parent_ids:
            parent_acts[cid] = d
        else:
            # find parent
            for h in hierarchies:
                if any(c.synset_id == cid for c in h.children):
                    child_acts.setdefault(h.parent.synset_id, {})[cid] = d
                    break

    # Estimate teacher vectors
    lda = LDAEstimator(shrinkage=float(cfg.get("geometry", {}).get("lda_shrinkage", 0.1)))
    est = ConceptVectorEstimator(lda, geom)
    parent_vecs = est.estimate_parent_vectors(parent_acts)
    child_vecs, child_deltas = est.estimate_child_deltas(parent_vecs, child_acts)

    # Angles
    stats = hierarchical_orthogonality(parent_vecs, child_deltas, geom, threshold_deg=float(cfg.get("eval", {}).get("angle_threshold_deg", 80)))
    # Collect raw angles and pair mapping for figures and subset metrics
    angles_deg = []
    angle_pairs = []
    for pid, deltas in child_deltas.items():
        if pid not in parent_vecs:
            continue
        p = parent_vecs[pid]
        for cid, d in deltas.items():
            if torch.norm(d) < 1e-8:
                continue
            ang = geom.causal_angle(p, d)
            ang_deg = float(torch.rad2deg(ang).item())
            angles_deg.append(ang_deg)
            angle_pairs.append({"parent_id": pid, "child_id": cid, "angle_deg": ang_deg})

    out = {
        "parent_vectors": {k: v.tolist() for k, v in parent_vecs.items()},
        "child_vectors": {pid: {cid: v.tolist() for cid, v in d.items()} for pid, d in child_vecs.items()},
        "child_deltas": {pid: {cid: v.tolist() for cid, v in d.items()} for pid, d in child_deltas.items()},
        "geometry_stats": geom.whitening_invariant_stats(),
        "angle_stats": stats,
        "angles_deg": angles_deg,
        "angle_pairs": angle_pairs,
    }
    with open(out_dir / "teacher_vectors.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", out_dir / "teacher_vectors.json")


if __name__ == "__main__":
    main()
