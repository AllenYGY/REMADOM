from __future__ import annotations
import torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self, hidden: int, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck)
        self.up = nn.Linear(bottleneck, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.relu(self.down(x)))

def attach_adapters_to_mlp(mlp: nn.Sequential, hidden: int, every: int = 1) -> nn.Sequential:
    """
    Insert adapters after each Linear-ReLU pair by frequency 'every'.
    """
    layers = []
    count = 0
    for m in mlp:
        layers.append(m)
        if isinstance(m, nn.ReLU):
            count += 1
            if count % every == 0:
                layers.append(Adapter(hidden))
    return nn.Sequential(*layers)

def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_adapters(m: nn.Module) -> None:
    for mod in m.modules():
        if isinstance(mod, Adapter):
            for p in mod.parameters():
                p.requires_grad = True