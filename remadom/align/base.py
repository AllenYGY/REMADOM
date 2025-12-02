from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch
from torch import nn, Tensor

class AlignmentHead(nn.Module):
    def __init__(self, weight: float = 1.0, name: str = "alignment"):
        super().__init__()
        self.weight = weight
        self.name = name

    def set_params(self, **kwargs) -> None:  # pragma: no cover - optional override
        if "weight" in kwargs:
            self.weight = float(kwargs["weight"])

    def forward(
        self,
        z_bio: Tensor,
        groups: Optional[Tensor] = None,
        graph: Optional[Any] = None,
        bridge: Optional[Any] = None,
        aux: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        raise NotImplementedError
