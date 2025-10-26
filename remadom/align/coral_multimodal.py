from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch import Tensor
from .base import AlignmentHead

def _covariance(X: Tensor, centered: bool = True) -> Tensor:
    if centered:
        X = X - X.mean(0, keepdim=True)
    n = X.shape[0]
    return (X.t() @ X) / max(1, (n - 1))

class CORALMultiHead(AlignmentHead):
    def __init__(self, weight: float = 1.0, centered: bool = True, pairs: Optional[List[tuple[str, str]]] = None):
        super().__init__(weight, "coral_multi")
        self.centered = centered
        self.pairs = pairs or []

    def forward(self, z_by_modality: Dict[str, Tensor], aux: Optional[Dict[str, Any]] = None) -> Tuple[Tensor, Dict[str, Any]]:
        loss = torch.tensor(0.0, device=next(iter(z_by_modality.values())).device)
        count = 0
        for (a, b) in self.pairs:
            if a not in z_by_modality or b not in z_by_modality:
                continue
            Ca = _covariance(z_by_modality[a], self.centered)
            Cb = _covariance(z_by_modality[b], self.centered)
            loss = loss + torch.nn.functional.mse_loss(Ca, Cb)
            count += 1
        if count > 0:
            loss = loss / count
        return self.weight * loss, {"pairs": count}