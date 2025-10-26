from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch import Tensor
from .base import AlignmentHead
from .solvers.gw_solver import GwSolver
from .costs import pairwise_cost

class GWMultiModalHead(AlignmentHead):
    def __init__(self, weight: float = 1.0, epsilon: float = 1e-3, fused_alpha: Optional[float] = None, pairs: Optional[List[tuple[str, str]]] = None):
        super().__init__(weight, "gw_multi")
        self.solver = GwSolver(epsilon=epsilon, fused_alpha=fused_alpha)
        self.pairs = pairs or []
        self.fused_alpha = fused_alpha

    def forward(self, z_by_modality: Dict[str, Tensor], aux: Optional[Dict[str, Any]] = None) -> Tuple[Tensor, Dict[str, Any]]:
        device = next(iter(z_by_modality.values())).device
        loss = torch.tensor(0.0, device=device)
        used = 0
        for (a, b) in self.pairs:
            if a not in z_by_modality or b not in z_by_modality:
                continue
            Za = z_by_modality[a]
            Zb = z_by_modality[b]
            D1 = torch.cdist(Za, Za)
            D2 = torch.cdist(Zb, Zb)
            mu = torch.full((Za.shape[0],), 1.0 / max(1, Za.shape[0]), device=device)
            nu = torch.full((Zb.shape[0],), 1.0 / max(1, Zb.shape[0]), device=device)
            if self.fused_alpha is None:
                P, _ = self.solver.solve_gw(D1, D2, mu, nu)
                # proxy
                loss = loss + (D1.mean() - 2 * (P * pairwise_cost(Za, Zb)).sum() + D2.mean())
            else:
                C_feat = pairwise_cost(Za, Zb)
                P, _ = self.solver.solve_fgw(D1, D2, C_feat, mu, nu)
                loss = loss + (C_feat * P).sum()
            used += 1
        if used > 0:
            loss = loss / used
        return self.weight * loss, {"pairs": used}