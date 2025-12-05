from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import AlignmentHead


class TemporalHead(AlignmentHead):
    """
    Encourages the latent space to respect temporal ordering by regressing
    the time label from the biological latent.
    """

    def __init__(self, weight: float = 1.0, *, group_key: str = "time"):
        super().__init__(weight, name="temporal")
        self.group_key = group_key

    def forward(
        self,
        z_bio: Tensor,
        groups: Optional[Tensor] = None,
        graph: Optional[Any] = None,
        bridge: Optional[Any] = None,
        aux: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if groups is None or z_bio is None or z_bio.numel() == 0:
            zero = torch.zeros((), device=z_bio.device if z_bio is not None else None)
            return zero, {}
        t = groups.float()
        t = (t - t.mean()) / (t.std() + 1e-6)

        # Solve z w = t with least squares; differentiable with respect to z.
        try:
            sol = torch.linalg.lstsq(z_bio, t).solution
        except Exception:
            pseudo_inv = torch.linalg.pinv(z_bio)
            sol = pseudo_inv @ t
        pred = z_bio @ sol
        loss = F.mse_loss(pred, t)

        corr = torch.tensor(0.0, device=z_bio.device)
        if pred.numel() > 1:
            cov = ((pred - pred.mean()) * (t - t.mean())).mean()
            corr = cov / (pred.std() * t.std() + 1e-6)
        logs: Dict[str, float] = {"temporal_corr": float(corr.detach().cpu())}
        return self.weight * loss, logs
