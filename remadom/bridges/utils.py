from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
from torch import Tensor
from .base import BridgeEdges

def bridge_to_mask(bridge: BridgeEdges, n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    s = bridge.src_idx.detach().cpu().numpy()
    d = bridge.dst_idx.detach().cpu().numpy()
    mask[s, d] = True
    mask[d, s] = True
    return mask

def diagnostics(bridge: BridgeEdges, n: int) -> Dict[str, Any]:
    s = bridge.src_idx.detach().cpu().numpy()
    d = bridge.dst_idx.detach().cpu().numpy()
    degree = np.zeros(n, dtype=int)
    np.add.at(degree, s, 1)
    np.add.at(degree, d, 1)
    return {
        "edges": int(len(s)),
        "degree_mean": float(degree.mean()),
        "degree_max": int(degree.max() if len(degree) else 0),
    }