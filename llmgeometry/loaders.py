"""
Convenience loaders/savers for common experiment artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import json
import torch

from .geometry import CausalGeometry
from .activations import ActivationCapture
from .concepts import load_concept_hierarchies


def load_geometry(path: str) -> CausalGeometry:
    """Load geometry.pt saved as {'Sigma': ..., 'W': ...} and return CausalGeometry."""
    p = Path(path)
    data = torch.load(p, map_location="cpu")
    d = data["W"].shape[0]
    geom = CausalGeometry(torch.eye(d), shrinkage=0.0)
    geom.Sigma = data["Sigma"].float()
    geom.W = data["W"].float()
    return geom


def load_teacher_vectors(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    """Load parent_vectors, child_vectors, child_deltas from teacher_vectors.json."""
    data = json.load(open(path))
    parents = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.get("parent_vectors", {}).items()}
    child_vecs = {
        pid: {cid: torch.tensor(v, dtype=torch.float32) for cid, v in d.items()}
        for pid, d in data.get("child_vectors", {}).items()
    }
    child_deltas = {
        pid: {cid: torch.tensor(v, dtype=torch.float32) for cid, v in d.items()}
        for pid, d in data.get("child_deltas", {}).items()
    }
    return parents, child_vecs, child_deltas


def load_hierarchies(path: str):
    return load_concept_hierarchies(path)


def load_activations(path: str):
    return ActivationCapture.load_hierarchical_activations(path)


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_yaml(path: str):
    import yaml
    return yaml.safe_load(open(path))

