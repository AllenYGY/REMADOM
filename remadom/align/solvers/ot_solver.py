from __future__ import annotations
from typing import Tuple, Dict, Any
import torch
from torch import Tensor

class OtSolver:
    def __init__(self, epsilon: float = 0.05, unbalanced: bool = False, tau1: float = 1.0, tau2: float = 1.0):
        self.epsilon = float(epsilon)
        self.unbalanced = bool(unbalanced)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)

    def solve(self, C: Tensor, mu: Tensor, nu: Tensor, max_iter: int = 500, tol: float = 1e-6) -> Tuple[Tensor, Dict[str, Any]]:
        eps = self.epsilon
        K = torch.exp(-C / (eps + 1e-12))
        u = torch.ones_like(mu) / mu.shape[0]
        v = torch.ones_like(nu) / nu.shape[0]
        iters = 0
        for iters in range(max_iter):
            u_prev = u
            Kv = K @ v + 1e-12
            u = mu / Kv
            Ku = K.t() @ u + 1e-12
            v = nu / Ku
            if torch.max(torch.abs(u - u_prev)) < tol:
                break
        P = torch.diag(u) @ K @ torch.diag(v)
        return P, {"iters": iters, "epsilon": eps}