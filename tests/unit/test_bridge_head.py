from __future__ import annotations

import torch

from remadom.adapters.bridge_head import BridgeHead, build_bridge_provider


def test_bridge_head_builds_edges() -> None:
    torch.manual_seed(0)
    z = torch.randn(30, 6)
    groups = torch.tensor([0] * 15 + [1] * 15)
    provider = build_bridge_provider("mnn", {"k": 5})
    head = BridgeHead(provider=provider, weight=0.5, group_key="dataset", normalize=True, max_edges=5)
    loss, logs = head(z, groups=groups)
    assert torch.isfinite(loss)
    assert logs["bridge_edges"] >= 0
