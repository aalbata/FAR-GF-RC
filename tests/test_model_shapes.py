import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from fargfrc.models.far_gf_rc import FARGFRC

def _model(n=17):
    torch.manual_seed(0)
    return FARGFRC(number_of_sensors=n, history_feature_dimension=7,
                   future_calendar_feature_dimension=4, input_steps=12, output_steps=12,
                   physical_adjacency=torch.rand(n, n))

def test_forward_shapes_and_finiteness():
    n = 17
    m = _model(n)
    h = torch.randn(2, 12, n, 7)
    h[..., 1] = torch.randint(0, 2, (2, 12, n)).float()
    h[..., 2] = torch.rand(2, 12, n)
    out = m(h, torch.randn(2, 12, 4))
    assert out["forecast_normalized"].shape == (2, 12, n)
    assert out["forecast_log_scale"].shape == (2, 12, n)
    assert out["reconstruction_normalized"].shape == (2, 12, n)
    assert out["reliability"].shape == (2, 12, n)
    assert torch.isfinite(out["forecast_normalized"]).all()
    assert float(out["forecast_log_scale"].min()) >= -6.0
    assert float(out["forecast_log_scale"].max()) <= 3.0

def test_adaptive_adjacency_row_stochastic():
    m = _model()
    A = m.adaptive_graph()
    assert torch.allclose(A.sum(-1), torch.ones(A.shape[0]), atol=1e-5)

def test_uniform_reliability_is_gating_noop():
    # With constant source reliability, the reliability-weighted renormalized
    # adjacency must equal the plain row-normalized adjacency.
    from fargfrc.models.far_gf_rc import ReliabilityGatedGraphLayer
    torch.manual_seed(0)
    A = torch.softmax(torch.randn(9, 9), dim=-1)
    H = torch.randn(3, 9, 8)
    r_const = torch.full((3, 9), 0.37)
    agg = ReliabilityGatedGraphLayer._aggregate
    out_const = agg(A, H, r_const)
    out_plain = agg(A, H, torch.ones(3, 9))
    assert torch.allclose(out_const, out_plain, atol=1e-6)
