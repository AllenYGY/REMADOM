from __future__ import annotations
import torch
from torch import Tensor
from .base import BridgeProvider, BridgeEdges

class MNNBridge(BridgeProvider):
    def __init__(self, k: int = 20, metric: str = "euclidean", backend: str = "faiss"):
        super().__init__()
        self.k = int(k)
        self.metric = metric
        self.backend = backend

    def build(self, Z: Tensor, groups: Tensor) -> BridgeEdges:
        uniq = torch.unique(groups)
        assert uniq.numel() == 2, "MNNBridge assumes exactly two groups"
        a = Z[groups == uniq[0]]
        b = Z[groups == uniq[1]]
        Da = torch.cdist(a, b) if self.metric == "euclidean" else 1.0 - torch.nn.functional.normalize(a, dim=-1) @ torch.nn.functional.normalize(b, dim=-1).t()
        # a->b
        nb = torch.topk(-Da, k=min(self.k, Da.shape[1]), dim=1).indices
        # b->a
        Db = Da.t()
        na = torch.topk(-Db, k=min(self.k, Db.shape[1]), dim=1).indices
        pairs_a = []
        pairs_b = []
        for i in range(a.shape[0]):
            for j in nb[i].tolist():
                # check mutual
                if i in na[j].tolist():
                    pairs_a.append(i)
                    pairs_b.append(j)
        src = torch.tensor(pairs_a, dtype=torch.long, device=Z.device)
        dst = torch.tensor(pairs_b, dtype=torch.long, device=Z.device)
        # map to global indices
        idx_a = torch.nonzero(groups == uniq[0], as_tuple=False).squeeze(1)
        idx_b = torch.nonzero(groups == uniq[1], as_tuple=False).squeeze(1)
        return BridgeEdges(src_idx=idx_a[src], dst_idx=idx_b[dst], weight=None)