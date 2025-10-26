from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


class ProductOfExperts(nn.Module):
    """
    Product-of-experts fusion for Gaussian posteriors with optional modality masks.
    """

    def __init__(self, temps: Optional[Sequence[float] | Dict[str, float]] = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.temps = temps
        self.eps = float(eps)

    def forward(
        self,
        mus: Iterable[Tensor],
        logvars: Iterable[Tensor],
        modality_names: Optional[Sequence[str]] = None,
        masks: Optional[Iterable[Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor]:
        mu_list = list(mus)
        lv_list = list(logvars)
        mask_list = list(masks) if masks is not None else [None] * len(mu_list)
        if not mu_list:
            raise ValueError("ProductOfExperts expects at least one posterior")
        if len(mu_list) != len(lv_list):
            raise ValueError("Mismatch between mean and logvar lists")
        if len(mask_list) != len(mu_list):
            raise ValueError("Mismatch between posteriors and masks")

        device = mu_list[0].device
        precision_sum = torch.zeros_like(mu_list[0], device=device)
        weighted_mean = torch.zeros_like(mu_list[0], device=device)
        for idx, (mu, lv, mask) in enumerate(zip(mu_list, lv_list, mask_list)):
            tau = torch.exp(-lv)  # precision
            if self.temps is not None:
                if isinstance(self.temps, Sequence):
                    weight = float(self.temps[idx]) if idx < len(self.temps) else 1.0
                else:
                    name = modality_names[idx] if modality_names is not None else str(idx)
                    weight = float(self.temps.get(name, 1.0))
                tau = tau * weight
            if mask is not None:
                mask_f = mask.to(dtype=tau.dtype, device=device).unsqueeze(-1)
                tau = tau * mask_f
                mu = mu * mask_f
            precision_sum = precision_sum + tau
            weighted_mean = weighted_mean + mu * tau
        # Prior expert N(0, I)
        prior_precision = torch.ones_like(precision_sum, device=device)
        precision_sum = precision_sum + prior_precision
        mu_poe = weighted_mean / precision_sum.clamp_min(self.eps)
        logvar_poe = -torch.log(precision_sum.clamp_min(self.eps))
        return mu_poe, logvar_poe
