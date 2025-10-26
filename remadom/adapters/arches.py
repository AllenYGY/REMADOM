from __future__ import annotations
from torch import nn, Tensor

class Adapter(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim)
        )

    def forward(self, h: Tensor) -> Tensor:
        return h + self.proj(h)