import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llmgeometry.geometry import CausalGeometry
from llmgeometry.validation import hierarchical_orthogonality, ratio_invariance_synthetic


def test_hierarchical_orthogonality_synthetic():
    d = 4
    U = torch.eye(d)
    geom = CausalGeometry(U, shrinkage=0.0)
    # Force identity geometry for a crisp synthetic test
    geom.Sigma = torch.eye(d)
    geom.W = torch.eye(d)
    parent_vectors = {"p": torch.tensor([1.0, 0.0, 0.0, 0.0])}
    # Two child deltas orthogonal to parent
    child_deltas = {
        "p": {
            "c1": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "c2": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        }
    }
    stats = hierarchical_orthogonality(parent_vectors, child_deltas, geom, threshold_deg=80.0)
    assert stats["n_pairs"] == 2
    assert stats["fraction_above_threshold"] == 1.0


def test_ratio_invariance_synthetic():
    d = 3
    U = torch.eye(d)
    geom = CausalGeometry(U, shrinkage=0.0)
    geom.Sigma = torch.eye(d)
    geom.W = torch.eye(d)
    p = torch.tensor([1.0, 0.0, 0.0])
    # Child vectors with same norm, orthogonal to parent
    c1 = torch.tensor([0.0, 1.0, 0.0])
    c2 = torch.tensor([0.0, -1.0, 0.0])
    res = ratio_invariance_synthetic(p, [c1, c2], geom, alphas=[0.5, 1.0, 2.0])
    # All KLs are ~0 because child norms are equal and orthogonal to parent
    for v in res["by_alpha"].values():
        assert float(v["kl_divergence"]) < 1e-6
