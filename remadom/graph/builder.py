from __future__ import annotations
from typing import Tuple, Optional, Dict
import torch
from torch import Tensor

class GraphBuilder:
    def __init__(self, k: int = 30, metric: str = "euclidean", backend: str = "faiss"):
        self.k = int(k)
        self.metric = metric
        self.backend = backend

    def build(self, Z: Tensor) -> Tuple[Tensor, Tensor]:
        D = torch.cdist(Z, Z)  # (n, n)
        idx = torch.topk(-D, k=min(self.k + 1, Z.shape[0]), dim=-1).indices[:, 1:]
        dist = D.gather(1, idx)
        return idx, dist

    def build_batch_balanced(self, Z: Tensor, batches: Tensor, k_by_batch: Optional[Dict[int, int]] = None) -> Tuple[Tensor, Tensor]:
        n = Z.shape[0]
        uniq = batches.unique().tolist()
        if k_by_batch is None:
            share = max(1, self.k // max(1, len(uniq) - 1))
            k_by_batch = {int(b): share for b in uniq}
        idx_list = []
        dist_list = []
        for i in range(n):
            ii = []
            dd = []
            for b, k in k_by_batch.items():
                if int(batches[i]) == b:
                    continue
                mask = (batches == b)
                Zb = Z[mask]
                if Zb.shape[0] == 0:
                    continue
                Db = torch.cdist(Z[i:i+1], Zb).squeeze(0)
                top = torch.topk(-Db, k=min(k, Db.shape[0])).indices
                glob = torch.nonzero(mask, as_tuple=False).squeeze(1)[top]
                ii.append(glob)
                dd.append(Db[top])
            ii = torch.cat(ii) if len(ii) else torch.empty(0, dtype=torch.long, device=Z.device)
            dd = torch.cat(dd) if len(dd) else torch.empty(0, dtype=Z.dtype, device=Z.device)
            idx_list.append(ii)
            dist_list.append(dd)
        maxk = max([x.numel() for x in idx_list]) if idx_list else 0
        idx_out = torch.full((n, maxk), -1, dtype=torch.long, device=Z.device)
        dist_out = torch.full((n, maxk), float("inf"), dtype=Z.dtype, device=Z.device)
        for i, (ii, dd) in enumerate(zip(idx_list, dist_list)):
            m = ii.numel()
            if m > 0:
                idx_out[i, :m] = ii
                dist_out[i, :m] = dd
        return idx_out, dist_out