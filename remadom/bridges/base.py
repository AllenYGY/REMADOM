from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor

@dataclass
class BridgeEdges:
    src_idx: Tensor
    dst_idx: Tensor
    weight: Optional[Tensor] = None

class BridgeProvider:
    def __init__(self): ...
    def build(self, Z: Tensor, groups: Tensor) -> BridgeEdges:
        raise NotImplementedError