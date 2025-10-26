from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor

class BridgeDictionary:
    def __init__(self, rank: Optional[int] = None, lam: float = 0.0):
        self.rank = rank
        self.lam = float(lam)
        self.W: Optional[Tensor] = None  # map from A->B

    def fit(self, X_a_bridge: Tensor, X_b_bridge: Tensor) -> None:
        # Simple ridge regression: solve min_W ||A W - B||^2 + lam ||W||^2
        A = X_a_bridge
        B = X_b_bridge
        if self.rank is not None and self.rank > 0 and self.rank < min(A.shape[1], B.shape[1]):
            # Low-rank via SVD truncate
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            Ur = U[:, :self.rank]
            Sr = S[:self.rank]
            Ar = Ur * Sr
            W = torch.linalg.lstsq(Ar, B).solution
            self.W = torch.linalg.lstsq(A, Ar @ W).solution
        else:
            AtA = A.t() @ A + self.lam * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
            AtB = A.t() @ B
            self.W = torch.linalg.solve(AtA, AtB)

    def map_a_to_b(self, X_a: Tensor) -> Tensor:
        assert self.W is not None, "BridgeDictionary not fit"
        return X_a @ self.W

    def map_b_to_a(self, X_b: Tensor) -> Tensor:
        assert self.W is not None, "BridgeDictionary not fit"
        # pseudo-inverse map
        Wpinv = torch.linalg.pinv(self.W)
        return X_b @ Wpinv