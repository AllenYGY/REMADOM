from __future__ import annotations

import torch

from remadom.adapters.bridge_head import BridgeHead, build_bridge_provider


def make_dummy_latents(n_a: int = 5, n_b: int = 5):
    torch.manual_seed(42)
    a = torch.randn(n_a, 4)
    b = a + 0.1 * torch.randn(n_a, 4)
    if n_b > n_a:
        extra = torch.randn(n_b - n_a, 4)
        b = torch.cat([b, extra], dim=0)
    z = torch.cat([a, b], dim=0)
    groups = torch.tensor([0] * n_a + [1] * n_b)
    return z, groups


def test_seeded_bridge_edges_nonzero():
    z, groups = make_dummy_latents()
    provider = build_bridge_provider(
        "seeded",
        {
            "seed_pairs": [
                (0, 0),
                (1, 1),
            ],
            "radius": 2,
        },
    )
    head = BridgeHead(provider=provider, weight=1.0, group_key="dataset", pairs=[(0, 1)])
    loss, logs = head(z, groups=groups)
    assert loss >= 0
    assert logs["bridge_edges"] > 0
