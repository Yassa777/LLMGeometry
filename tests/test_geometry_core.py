import math
import os
import sys
import torch

# Add repo root to path for in-repo testing without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llmgeometry.geometry import CausalGeometry


def test_whiten_shapes_and_identity():
    torch.manual_seed(0)
    V, d = 64, 8
    U = torch.randn(V, d)
    geom = CausalGeometry(U, shrinkage=1e-6)

    x = torch.randn(10, d)
    xw = geom.whiten(x)
    xr = geom.unwhiten(xw)

    assert xw.shape == x.shape
    assert xr.shape == x.shape

    res = geom.test_linear_identity(x)
    assert math.isfinite(res["identity_mse"]) and math.isfinite(res["identity_ev"]) 
    assert res["identity_ev"] > 0.99


def test_causal_angle_properties():
    torch.manual_seed(0)
    V, d = 80, 6
    U = torch.randn(V, d)
    geom = CausalGeometry(U)

    a = torch.randn(d)
    assert torch.allclose(geom.causal_angle(a, 2.0 * a), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(geom.causal_angle(a, -a), torch.tensor(math.pi), atol=1e-6)

    aw = geom.whiten(a)
    r = torch.randn_like(aw)
    bw = r - (r @ aw) / (aw @ aw + 1e-8) * aw
    b = geom.unwhiten(bw)
    ang = geom.causal_angle(a, b)
    assert abs(ang.item() - math.pi / 2) < 1e-2

    z = torch.zeros_like(a)
    assert torch.isnan(geom.causal_angle(a, z)) or not torch.isfinite(geom.causal_angle(a, z))
