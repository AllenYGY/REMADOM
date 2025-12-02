from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch
from torch import Tensor
from .base import AlignmentHead
from .costs import pairwise_cost
from .solvers.gw_solver import GwSolver

class GWHead(AlignmentHead):
    def __init__(
        self,
        weight: float = 1.0,
        epsilon: float = 1e-3,
        fused_alpha: Optional[float] = None,
        group_key: str = "batch",
    ):
        super().__init__(weight, "gw")
        self.epsilon = float(epsilon)
        self.fused_alpha = fused_alpha
        self.group_key = group_key
        self.solver = GwSolver(epsilon=self.epsilon, fused_alpha=fused_alpha)

    def set_params(self, **kwargs) -> None:
        super().set_params(**kwargs)
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

        total = torch.tensor(0.0, device=z_bio.device)
        count = 0
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a = z_bio[groups == uniq[i]]
                b = z_bio[groups == uniq[j]]
                if a.numel() == 0 or b.numel() == 0:
                    continue
                D1 = torch.cdist(a, a)
                D2 = torch.cdist(b, b)
                mu = torch.full((a.shape[0],), 1.0 / max(1, a.shape[0]), device=z_bio.device)
                nu = torch.full((b.shape[0],), 1.0 / max(1, b.shape[0]), device=z_bio.device)
                if self.fused_alpha is None:
                    P, _ = self.solver.solve_gw(D1, D2, mu, nu)
                    loss = (D1.mean() - 2 * (P * pairwise_cost(a, b)).sum() + D2.mean())
                else:
                    C_feat = pairwise_cost(a, b)
                    P, _ = self.solver.solve_fgw(D1, D2, C_feat, mu, nu)
                    loss = (C_feat * P).sum()
                total = total + loss
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=z_bio.device), {"pairs": 0, "epsilon": self.epsilon, "fused_alpha": self.fused_alpha}
        avg = total / count
        logs = {"pairs": count, "epsilon": self.epsilon, "fused_alpha": self.fused_alpha}
        return self.weight * avg, logs
