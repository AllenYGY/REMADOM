from __future__ import annotations
import torch
from torch import Tensor

def pairwise_cost(Z_a: Tensor, Z_b: Tensor, metric: str = "euclidean") -> Tensor:
    if metric == "euclidean":
        return torch.cdist(Z_a, Z_b)
    if metric == "cosine":
        Za = torch.nn.functional.normalize(Z_a, dim=-1)
        Zb = torch.nn.functional.normalize(Z_b, dim=-1)
        return 1.0 - Za @ Zb.t()
    raise ValueError(f"Unknown metric {metric}")