from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch import Tensor
from .base import AlignmentHead
from .mmd import _rbf_mmd2

class MMDMultiHead(AlignmentHead):
    def __init__(self, weight: float = 1.0, kernel: str = "rbf", pairs: Optional[List[tuple[str, str]]] = None, bandwidth: float = 1.0):
        super().__init__(weight, "mmd_multi")
        self.kernel = kernel
        self.pairs = pairs or []
        self.bandwidth = bandwidth

    def forward(self, z_by_modality: Dict[str, Tensor], aux: Optional[Dict[str, Any]] = None) -> Tuple[Tensor, Dict[str, Any]]:
        loss = torch.tensor(0.0, device=next(iter(z_by_modality.values())).device)
        count = 0
        for (a, b) in self.pairs:
            if a not in z_by_modality or b not in z_by_modality:
                continue
            za, zb = z_by_modality[a], z_by_modality[b]
            if za.numel() == 0 or zb.numel() == 0:
                continue
            loss = loss + _rbf_mmd2(za, zb, self.bandwidth)
            count += 1
        if count > 0:
            loss = loss / count
        return self.weight * loss, {"pairs": count}