from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch
from torch import Tensor
from .base import AlignmentHead
from .costs import pairwise_cost
from .solvers.gw_solver import GwSolver

class GWHead(AlignmentHead):
    def __init__(self, weight: float = 1.0, epsilon: float = 1e-3, fused_alpha: Optional[float] = None):
        super().__init__(weight, "gw")
        self.epsilon = float(epsilon)
        self.fused_alpha = fused_alpha
        self.solver = GwSolver(epsilon=self.epsilon, fused_alpha=fused_alpha)

    def set_params(self, **kwargs) -> None:
        changed = False
        if "epsilon" in kwargs:
            self.epsilon = float(kwargs["epsilon"])
            changed = True
        if "fused_alpha" in kwargs:
            self.fused_alpha = None if kwargs["fused_alpha"] is None else float(kwargs["fused_alpha"])
            changed = True
        if changed:
            self.solver = GwSolver(epsilon=self.epsilon, fused_alpha=self.fused_alpha)

    def forward(
        self,
        z_bio: Tensor,
        groups: Optional[Tensor] = None,
        graph: Optional[Any] = None,
        bridge: Optional[Any] = None,
        aux: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        assert groups is not None, "GWHead expects groups"
        uniq = torch.unique(groups)
        if uniq.numel() < 2:
            return torch.tensor(0.0, device=z_bio.device), {}
        a = z_bio[groups == uniq[0]]
        b = z_bio[groups == uniq[1]]
        D1 = torch.cdist(a, a)
        D2 = torch.cdist(b, b)
        mu = torch.full((a.shape[0],), 1.0 / max(1, a.shape[0]), device=z_bio.device)
        nu = torch.full((b.shape[0],), 1.0 / max(1, b.shape[0]), device=z_bio.device)
        if self.fused_alpha is None:
            P, logs = self.solver.solve_gw(D1, D2, mu, nu)
            loss = (D1.mean() - 2 * (P * pairwise_cost(a, b)).sum() + D2.mean())
            logs.update({"epsilon": self.epsilon, "fused_alpha": self.fused_alpha})
            return self.weight * loss, logs
        C_feat = pairwise_cost(a, b)
        P, logs = self.solver.solve_fgw(D1, D2, C_feat, mu, nu)
        loss = (C_feat * P).sum()
        logs.update({"epsilon": self.epsilon, "fused_alpha": self.fused_alpha})
        return self.weight * loss, logs