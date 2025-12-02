from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .base import BridgeEdges, BridgeProvider


class SeededBridge(BridgeProvider):
    """Bridge provider that relies on pre-defined seed pairs, with optional neighbourhood expansion."""

    def __init__(self, seed_pairs: Optional[Sequence[Tuple[int, int]]] = None, radius: int = 0):
        super().__init__()
        self.seed_pairs: List[Tuple[int, int]] = [(int(a), int(b)) for a, b in seed_pairs] if seed_pairs else []
        self.radius = max(0, int(radius))

    def build(self, Z: Tensor, groups: Tensor) -> BridgeEdges:
        uniq = torch.unique(groups)
        assert uniq.numel() == 2, "SeededBridge expects exactly two groups"
        idx_a = torch.nonzero(groups == uniq[0], as_tuple=False).squeeze(1)
        idx_b = torch.nonzero(groups == uniq[1], as_tuple=False).squeeze(1)
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            return BridgeEdges(src_idx=torch.empty(0, dtype=torch.long, device=Z.device), dst_idx=torch.empty(0, dtype=torch.long, device=Z.device))

        src_indices: List[int] = []
        dst_indices: List[int] = []

        if self.seed_pairs:
            for sa, sb in self.seed_pairs:
                if 0 <= sa < idx_a.numel() and 0 <= sb < idx_b.numel():
                    src_indices.append(int(idx_a[sa]))
                    dst_indices.append(int(idx_b[sb]))

        if self.radius > 0 and src_indices:
            src_tensor = Z[idx_a]
            dst_tensor = Z[idx_b]
            for sa, sb in self.seed_pairs:
                if 0 <= sa < idx_a.numel() and 0 <= sb < idx_b.numel():
                    anchor_a = src_tensor[sa].unsqueeze(0)
                    anchor_b = dst_tensor[sb].unsqueeze(0)
                    dist_a = torch.cdist(anchor_a, src_tensor).squeeze(0)
                    dist_b = torch.cdist(anchor_b, dst_tensor).squeeze(0)
                    nn_a = torch.topk(-dist_a, k=min(self.radius, dist_a.numel()), dim=0).indices
                    nn_b = torch.topk(-dist_b, k=min(self.radius, dist_b.numel()), dim=0).indices
                    for na in nn_a.tolist():
                        for nb in nn_b.tolist():
                            src_indices.append(int(idx_a[na]))
                            dst_indices.append(int(idx_b[nb]))

        if not src_indices:
            return BridgeEdges(src_idx=torch.empty(0, dtype=torch.long, device=Z.device), dst_idx=torch.empty(0, dtype=torch.long, device=Z.device))

        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=Z.device)
        dst_tensor = torch.tensor(dst_indices, dtype=torch.long, device=Z.device)
        src_tensor, dst_tensor = _pair_unique(src_tensor, dst_tensor)
        return BridgeEdges(src_idx=src_tensor, dst_idx=dst_tensor)


def _pair_unique(src: Tensor, dst: Tensor) -> Tuple[Tensor, Tensor]:
    if src.numel() == 0:
        return src, dst
    seen: dict[int, int] = {}
    for i, value in enumerate(dst.tolist()):
        if value not in seen:
            seen[value] = i
    indices = torch.tensor(sorted(seen.values()), dtype=torch.long, device=src.device)
    return src[indices], dst[indices]
