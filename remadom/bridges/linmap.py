from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import BridgeEdges, BridgeProvider


class LinearMapBridge(BridgeProvider):
    """
    Bridge provider that fits a linear map between two cohorts and returns edges via nearest mapping.
    """

    def __init__(self, lam: float = 0.0, bridge_size: Optional[int] = None):
        super().__init__()
        self.lam = float(lam)
        self.bridge_size = bridge_size

    def build(self, Z: Tensor, groups: Tensor) -> BridgeEdges:
        uniq = torch.unique(groups)
        assert uniq.numel() == 2, "LinearMapBridge expects exactly two cohorts"
        idx_a = torch.nonzero(groups == uniq[0], as_tuple=False).squeeze(1)
        idx_b = torch.nonzero(groups == uniq[1], as_tuple=False).squeeze(1)
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            return BridgeEdges(src_idx=torch.empty(0, dtype=torch.long, device=Z.device), dst_idx=torch.empty(0, dtype=torch.long, device=Z.device))

        bridge_count = min(idx_a.numel(), idx_b.numel())
        if self.bridge_size is not None:
            bridge_count = min(bridge_count, int(self.bridge_size))
        if bridge_count == 0:
            return BridgeEdges(src_idx=torch.empty(0, dtype=torch.long, device=Z.device), dst_idx=torch.empty(0, dtype=torch.long, device=Z.device))

        z_a = Z[idx_a]
        z_b = Z[idx_b]

        # Fit linear map A->B with ridge
        bridge_a = z_a[:bridge_count]
        bridge_b = z_b[:bridge_count]
        A = bridge_a
        B = bridge_b
        lamI = self.lam * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        lhs = A.t() @ A + lamI
        rhs = A.t() @ B
        try:
            W = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            W = torch.linalg.lstsq(lhs, rhs).solution

        mapped = z_a @ W
        dists = torch.cdist(mapped, z_b)
        dst_choice = torch.argmin(dists, dim=1)
        src_idx = idx_a
        dst_idx = idx_b[dst_choice]
        src_idx, dst_idx = _deduplicate(src_idx, dst_idx)
        return BridgeEdges(src_idx=src_idx, dst_idx=dst_idx)


def _deduplicate(src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
    if src.numel() == 0:
        return src, dst
    seen = {}
    for i, v in enumerate(dst.tolist()):
        if v not in seen:
            seen[v] = i
    keep = torch.tensor(sorted(seen.values()), dtype=torch.long, device=src.device)
    return src[keep], dst[keep]
