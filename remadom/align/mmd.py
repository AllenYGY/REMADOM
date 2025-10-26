from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch
from torch import Tensor
from .base import AlignmentHead

def _rbf_mmd2(x: Tensor, y: Tensor, bandwidth: float = 1.0) -> Tensor:
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    kxx = torch.exp(-xx / (2 * bandwidth**2))
    kyy = torch.exp(-yy / (2 * bandwidth**2))
    kxy = torch.exp(-xy / (2 * bandwidth**2))
    m = x.shape[0]
    n = y.shape[0]
    mmd2 = kxx.sum() / (m * m) + kyy.sum() / (n * n) - 2 * kxy.sum() / (m * n)
    return mmd2

class MMDHead(AlignmentHead):
    def __init__(self, weight: float = 1.0, kernel: str = "rbf", group_key: str = "batch", bandwidth: float = 1.0):
        super().__init__(weight, "mmd")
        self.kernel = kernel
        self.group_key = group_key
        self.bandwidth = bandwidth

    def forward(
        self,
        z_bio: Tensor,
        groups: Optional[Tensor] = None,
        graph: Optional[Any] = None,
        bridge: Optional[Any] = None,
        aux: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        assert groups is not None, "MMDHead requires groups"
        unique = torch.unique(groups)
        loss = torch.tensor(0.0, device=z_bio.device)
        count = 0
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                a = z_bio[groups == unique[i]]
                b = z_bio[groups == unique[j]]
                if a.numel() == 0 or b.numel() == 0:
                    continue
                loss = loss + _rbf_mmd2(a, b, self.bandwidth)
                count += 1
        if count > 0:
            loss = loss / count
        return self.weight * loss, {"mmd_pairs": count}