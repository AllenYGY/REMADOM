from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from torch import Tensor
from .base import AlignmentHead
from .costs import pairwise_cost
from .solvers.ot_solver import OtSolver

class SinkhornHead(AlignmentHead):
    def __init__(self, weight: float = 1.0, epsilon: float = 0.05, group_key: str = "batch", metric: str = "euclidean"):
        super().__init__(weight, "sinkhorn")
        self.epsilon = float(epsilon)
        self.group_key = group_key
        self.metric = metric
        self.solver = OtSolver(epsilon=self.epsilon)

    def set_params(self, **kwargs) -> None:
        super().set_params(**kwargs)
        if "epsilon" in kwargs:
            self.epsilon = float(kwargs["epsilon"])
            self.solver = OtSolver(epsilon=self.epsilon)

    def forward(
        self,
        z_bio: Tensor,
        groups: Optional[Tensor] = None,
        graph: Optional[Any] = None,
        bridge: Optional[Any] = None,
        aux: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        assert groups is not None, "SinkhornHead requires groups"
        uniq = torch.unique(groups)
        if uniq.numel() < 2:
            return torch.tensor(0.0, device=z_bio.device), {}

        total_cost = torch.tensor(0.0, device=z_bio.device)
        pair_logs: Dict[str, Any] = {"pairs": 0, "epsilon": self.epsilon}

        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a = z_bio[groups == uniq[i]]
                b = z_bio[groups == uniq[j]]
                if a.numel() == 0 or b.numel() == 0:
                    continue
                C = pairwise_cost(a, b, self.metric)
                mu = torch.full((a.shape[0],), 1.0 / max(1, a.shape[0]), device=z_bio.device)
                nu = torch.full((b.shape[0],), 1.0 / max(1, b.shape[0]), device=z_bio.device)
                P, logs = self.solver.solve(C, mu, nu)
                cost = (C * P).sum()
                total_cost = total_cost + cost
                pair_logs["pairs"] += 1
        if pair_logs["pairs"] == 0:
            return torch.tensor(0.0, device=z_bio.device), pair_logs
        avg_cost = total_cost / pair_logs["pairs"]
        pair_logs["ot_cost"] = float(avg_cost.detach().cpu())
        return self.weight * avg_cost, pair_logs
