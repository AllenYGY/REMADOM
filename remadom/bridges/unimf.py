from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor

class BridgeUINMF:
    def __init__(self, k_shared: int, lam: float = 0.0, max_iter: int = 500):
        self.k = int(k_shared)
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.Wa: Optional[Tensor] = None
        self.Wb: Optional[Tensor] = None
        self.H: Optional[Tensor] = None

    def fit(self, X_a_bridge: Tensor, X_b_bridge: Tensor) -> None:
        # Simplified alternating ridge factorization:
        n = X_a_bridge.shape[0]
        H = torch.randn((n, self.k), device=X_a_bridge.device, dtype=X_a_bridge.dtype)
        Wa = torch.randn((X_a_bridge.shape[1], self.k), device=X_a_bridge.device, dtype=X_a_bridge.dtype)
        Wb = torch.randn((X_b_bridge.shape[1], self.k), device=X_b_bridge.device, dtype=X_b_bridge.dtype)
        for _ in range(self.max_iter):
            # Update H: solve (Wa^T Wa + Wb^T Wb + lam I) H = Wa^T Xa + Wb^T Xb
            A = Wa.t() @ Wa + Wb.t() @ Wb + self.lam * torch.eye(self.k, device=H.device, dtype=H.dtype)
            B = X_a_bridge @ Wa + X_b_bridge @ Wb
            H = torch.linalg.lstsq(A, B.t()).solution.t()
            # Update Wa, Wb: ridge regression per feature
            Wa = torch.linalg.lstsq(H, X_a_bridge).solution.t()
            Wb = torch.linalg.lstsq(H, X_b_bridge).solution.t()
        self.Wa, self.Wb, self.H = Wa, Wb, H

    def encode_a(self, X_a: Tensor) -> Tensor:
        assert self.Wa is not None
        # project: H ≈ X_a W_a (least squares)
        return X_a @ torch.linalg.pinv(self.Wa)

    def encode_b(self, X_b: Tensor) -> Tensor:
        assert self.Wb is not None
        return X_b @ torch.linalg.pinv(self.Wb)