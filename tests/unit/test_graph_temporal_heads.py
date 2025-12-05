import torch

from remadom.align.graph import GraphHead
from remadom.align.temporal import TemporalHead


def test_graph_head_returns_loss_and_logs():
    torch.manual_seed(0)
    z = torch.randn(32, 4)
    head = GraphHead(weight=1.0, k=5, lam=1e-2)
    loss, logs = head(z)
    assert loss >= 0
    assert "graph_edges" in logs
    assert logs["graph_edges"] > 0


def test_temporal_head_aligns_time():
    t = torch.linspace(0, 1, 20)
    z = torch.stack([t, t * 0 + 0.1, t * 0 + 0.2], dim=1)
    head = TemporalHead(weight=1.0, group_key="time")
    loss, logs = head(z, groups=t)
    assert loss < 0.05
    assert "temporal_corr" in logs
