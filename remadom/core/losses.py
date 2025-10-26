from __future__ import annotations
from typing import Dict
import torch
from torch import Tensor

def kl_gaussian(mu: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1).mean()

def nll_nb(x: Tensor, params: Dict[str, Tensor]) -> Tensor:
    mu = params["mu"].clamp_min(1e-8)
    theta = params["theta"].clamp_min(1e-8)
    lgamma = torch.lgamma
    t1 = lgamma(theta + x) - lgamma(theta) - lgamma(x + 1.0)
    t2 = theta * torch.log(theta / (theta + mu))
    t3 = x * torch.log(mu / (theta + mu))
    return (-(t1 + t2 + t3)).mean()

def nll_zinb(x: Tensor, params: Dict[str, Tensor]) -> Tensor:
    mu = params["mu"].clamp_min(1e-8)
    theta = params["theta"].clamp_min(1e-8)
    pi = params.get("pi", None)
    lgamma = torch.lgamma
    t1 = lgamma(theta + x) - lgamma(theta) - lgamma(x + 1.0)
    t2 = theta * (torch.log(theta) - torch.log(theta + mu))
    t3 = x * (torch.log(mu) - torch.log(theta + mu))
    nb = -(t1 + t2 + t3)
    if pi is None:
        return nb.mean()
    zero_case = -torch.log(pi + (1 - pi) * torch.exp(-nb) + 1e-8)
    nonzero_case = -torch.log(1 - pi + 1e-8) + nb
    loss = torch.where(x < 1e-8, zero_case, nonzero_case)
    return loss.mean()

def nll_bernoulli(x: Tensor, params: Dict[str, Tensor]) -> Tensor:
    logits = params["logits"]
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="mean")