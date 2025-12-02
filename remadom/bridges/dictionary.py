from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor

class BridgeDictionary:
    def __init__(self, rank: Optional[int] = None, lam: float = 0.0):
        self.rank = rank
        self.lam = float(lam)
        self.W: Optional[Tensor] = None  # map from A->B

    def fit(self, X_a_bridge: Tensor, X_b_bridge: Tensor) -> None:
        # Simple ridge regression: solve min_W ||A W - B||^2 + lam ||W||^2
        A = X_a_bridge
        B = X_b_bridge
        if self.rank is not None and self.rank > 0 and self.rank < min(A.shape[1], B.shape[1]):
            # Low-rank via SVD truncate
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            Ur = U[:, :self.rank]
            Sr = S[:self.rank]
            Ar = Ur * Sr
            W = torch.linalg.lstsq(Ar, B).solution
            self.W = torch.linalg.lstsq(A, Ar @ W).solution
        else:
            AtA = A.t() @ A + self.lam * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
            AtB = A.t() @ B
            try:
                self.W = torch.linalg.solve(AtA, AtB)
            except RuntimeError:
                # Fallback to least squares if singular
                self.W = torch.linalg.lstsq(AtA, AtB).solution

    def map_a_to_b(self, X_a: Tensor) -> Tensor:
        assert self.W is not None, "BridgeDictionary not fit"
        return X_a @ self.W

    def map_b_to_a(self, X_b: Tensor) -> Tensor:
        assert self.W is not None, "BridgeDictionary not fit"
        # pseudo-inverse map
        Wpinv = torch.linalg.pinv(self.W)
        return X_b @ Wpinv


class DictionaryBridgeProvider:
    """Bridge provider that fits a linear dictionary between cohorts and links nearest neighbours."""

    def __init__(self, bridge_size: Optional[int] = None, lam: float = 0.0):
        self.bridge_size = bridge_size
        self.lam = lam

    def build(self, Z: Tensor, groups: Tensor) -> "BridgeEdges":
        from .base import BridgeEdges  # avoid circular import

        uniq = torch.unique(groups)
        assert uniq.numel() == 2, "DictionaryBridgeProvider expects exactly two cohorts"
        idx_a = torch.nonzero(groups == uniq[0], as_tuple=False).squeeze(1)
        idx_b = torch.nonzero(groups == uniq[1], as_tuple=False).squeeze(1)
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            return BridgeEdges(src_idx=torch.empty(0, dtype=torch.long, device=Z.device), dst_idx=torch.empty(0, dtype=torch.long, device=Z.device))

        bridge_count = min(idx_a.numel(), idx_b.numel())
        if self.bridge_size is not None:
            bridge_count = min(bridge_count, int(self.bridge_size))
        if bridge_count == 0:
            return BridgeEdges(src_idx=torch.empty(0, dtype=torch.long, device=Z.device), dst_idx=torch.empty(0, dtype=torch.long, device=Z.device))

        bridge_a = Z[idx_a[:bridge_count]]
        bridge_b = Z[idx_b[:bridge_count]]
        dictionary = BridgeDictionary(lam=self.lam)
        dictionary.fit(bridge_a, bridge_b)

        mapped = dictionary.map_a_to_b(Z[idx_a])
        dist = torch.cdist(mapped, Z[idx_b])
        # choose best match for each source
        dst_choice = torch.argmin(dist, dim=1)
        src_idx = idx_a
        dst_idx = idx_b[dst_choice]

        # ensure mutual uniqueness by optional filtering
        src_idx, dst_idx = self._deduplicate(src_idx, dst_idx)
        return BridgeEdges(src_idx=src_idx, dst_idx=dst_idx)

    @staticmethod
    def _deduplicate(src: Tensor, dst: Tensor) -> Tuple[Tensor, Tensor]:
        if src.numel() == 0:
            return src, dst
        seen: dict[int, int] = {}
        for i, value in enumerate(dst.tolist()):
            if value not in seen:
                seen[value] = i
        indices = torch.tensor(sorted(seen.values()), dtype=torch.long, device=src.device)
        return src[indices], dst[indices]
