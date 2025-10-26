from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import torch
from torch import Tensor
from ..costs import pairwise_cost
from ...utils.capability import has_pot

class GwSolver:
    def __init__(self, epsilon: float = 1e-3, fused_alpha: float | None = None):
        self.epsilon = float(epsilon)
        self.fused_alpha = fused_alpha

    def _sliced_pairwise_dist(self, Z: Tensor, n_proj: int = 64, seed: int = 0) -> Tensor:
        g = torch.Generator(device=Z.device).manual_seed(seed)
        d = Z.shape[1]
        dirs = torch.randn((d, n_proj), generator=g, device=Z.device)
        dirs = torch.nn.functional.normalize(dirs, dim=0)
        proj = Z @ dirs  # (n, n_proj)
        # 1D |x_i - x_j|
        D = torch.cdist(proj.t(), proj.t(), p=2).mean(0)  # rough surrogate
        return D

    def solve_gw(self, D1: Tensor, D2: Tensor, mu: Tensor, nu: Tensor, max_iter: int = 100) -> Tuple[Tensor, Dict[str, Any]]:
        if has_pot():
            import ot.gromov as og  # type: ignore
            D1_np = D1.detach().cpu().numpy()
            D2_np = D2.detach().cpu().numpy()
            mu_np = mu.detach().cpu().numpy()
            nu_np = nu.detach().cpu().numpy()
            P_np = og.gromov_wasserstein(D1_np, D2_np, mu_np, nu_np, 'square_loss', epsilon=self.epsilon, max_iter=max_iter)
            P = torch.tensor(P_np, device=D1.device, dtype=D1.dtype)
            return P, {"gw_solver": "pot", "iters": max_iter}
        # sliced fallback: build approximate D1/D2 from embeddings is expected externally
        # Here we just return uniform coupling
        P = torch.outer(mu, nu)
        return P, {"gw_solver": "sliced_fallback", "iters": 0}

    def solve_fgw(
        self,
        D1: Tensor,
        D2: Tensor,
        C_feat: Tensor,
        mu: Tensor,
        nu: Tensor,
        max_iter: int = 100,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if has_pot():
            import ot.gromov as og  # type: ignore
            D1_np = D1.detach().cpu().numpy()
            D2_np = D2.detach().cpu().numpy()
            C_np = C_feat.detach().cpu().numpy()
            mu_np = mu.detach().cpu().numpy()
            nu_np = nu.detach().cpu().numpy()
            alpha = 0.5 if self.fused_alpha is None else float(self.fused_alpha)
            P_np = og.fused_gromov_wasserstein(C_np, D1_np, D2_np, mu_np, nu_np, 'square_loss', alpha=alpha, epsilon=self.epsilon, max_iter=max_iter)
            P = torch.tensor(P_np, device=D1.device, dtype=D1.dtype)
            return P, {"fgw_solver": "pot", "iters": max_iter, "alpha": alpha}
        P = torch.outer(mu, nu)
        return P, {"fgw_solver": "sliced_fallback", "iters": 0, "alpha": self.fused_alpha if self.fused_alpha is not None else 0.5}
