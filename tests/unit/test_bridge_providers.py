from __future__ import annotations

import torch

from remadom.bridges.dictionary import DictionaryBridgeProvider
from remadom.bridges.linmap import LinearMapBridge
from remadom.bridges.mnn import MNNBridge
from remadom.bridges.seeded import SeededBridge


def _toy_data():
    # two clusters offset in latent space
    z_a = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    z_b = torch.tensor([[1.0, 1.0], [1.1, 1.0], [0.9, 1.0]])
    z = torch.cat([z_a, z_b], dim=0)
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    return z, groups


def test_mnn_bridge_edges():
    z, groups = _toy_data()
    provider = MNNBridge(k=2, metric="euclidean", backend="numpy")
    edges = provider.build(z, groups)
    assert edges.src_idx.numel() > 0
    assert edges.src_idx.shape == edges.dst_idx.shape


def test_seeded_bridge_edges():
    z, groups = _toy_data()
    provider = SeededBridge(seed_pairs=[(0, 0)], radius=1)
    edges = provider.build(z, groups)
    assert edges.src_idx.numel() > 0
    assert edges.dst_idx.numel() == edges.src_idx.numel()


def test_dictionary_bridge_edges():
    z, groups = _toy_data()
    provider = DictionaryBridgeProvider(bridge_size=2, lam=0.0)
    edges = provider.build(z, groups)
    assert edges.src_idx.numel() > 0
    assert edges.dst_idx.numel() == edges.src_idx.numel()


def test_linmap_bridge_edges():
    z, groups = _toy_data()
    provider = LinearMapBridge(lam=0.0, bridge_size=2)
    edges = provider.build(z, groups)
    assert edges.src_idx.numel() > 0
    assert edges.dst_idx.numel() == edges.src_idx.numel()
