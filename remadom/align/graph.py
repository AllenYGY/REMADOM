from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import torch
from torch import Tensor

from .base import AlignmentHead
from ..graph.builder import GraphBuilder
from ..graph.laplacian import GraphRegularizer


class GraphHead(AlignmentHead):
    """
    Laplacian smoothing head on the biological latent. Builds a kNN graph
    within the batch and applies a Laplacian penalty.
    """

    def __init__(
        self,
        weight: float = 1.0,
        *,
        k: int = 15,
        metric: str = "euclidean",
        normalized: bool = True,
        lam: float = 1e-3,
    ):
        super().__init__(weight, name="graph")
        self.k = int(k)
        self.metric = metric
        self.normalized = bool(normalized)
        self.reg = GraphRegularizer(lam=lam, normalized=normalized)
        self.builder = GraphBuilder(k=self.k, metric=self.metric)

    def set_params(self, **kwargs) -> None:  # pragma: no cover - optional schedule support
        super().set_params(**kwargs)
        if "k" in kwargs:
            self.k = int(kwargs["k"])
            self.builder.k = self.k
        if "lam" in kwargs:
            self.reg.lam = float(kwargs["lam"])

    def forward(
        self,
        z_bio: Tensor,
        groups: Optional[Tensor] = None,
        graph: Optional[Any] = None,
        bridge: Optional[Any] = None,
        aux: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if z_bio is None or z_bio.numel() == 0:
            zero = torch.zeros((), device=z_bio.device if z_bio is not None else None)
            return zero, {}
        idx, dist = self.builder.build(z_bio.detach())
        if idx.numel() == 0:
            return torch.zeros((), device=z_bio.device), {}
        weights = torch.exp(-dist)
        loss, logs = self.reg(z_bio, idx, weights, num_nodes=z_bio.shape[0])
        logs["graph_k"] = self.k
        logs["graph_edges"] = int((idx >= 0).sum().item())
        return self.weight * loss, logs
