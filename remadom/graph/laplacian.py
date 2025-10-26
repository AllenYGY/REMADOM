from __future__ import annotations
from typing import Tuple, Dict, Any
import torch
from torch import nn, Tensor

class GraphRegularizer(nn.Module):
    def __init__(self, lam: float = 1e-3, normalized: bool = True):
        super().__init__()
        self.lam = float(lam)
        self.normalized = bool(normalized)

    def forward(self, Z: Tensor, indices: Tensor, weights: Tensor, num_nodes: int) -> Tuple[Tensor, Dict[str, Any]]:
        # Build sparse adjacency
        device = Z.device
        i = torch.arange(num_nodes, device=device).unsqueeze(1).expand_as(indices)
        rows = i.reshape(-1)
        cols = indices.reshape(-1)
        vals = weights.reshape(-1)
        mask = cols >= 0
        rows = rows[mask]
        cols = cols[mask]
        vals = vals[mask]
        A = torch.sparse_coo_tensor(torch.vstack([rows, cols]), vals, size=(num_nodes, num_nodes))
        A = (A + A.transpose(0, 1)) * 0.5  # symmetrize
        deg = torch.sparse.sum(A, dim=1).to_dense()
        if self.normalized:
            d_inv_sqrt = (deg + 1e-8).pow(-0.5)
            # L_norm = I - D^{-1/2} A D^{-1/2}
            Z_norm = d_inv_sqrt.unsqueeze(1) * Z
            AZ = torch.sparse.mm(A, Z_norm)
            smooth = (Z_norm * deg.unsqueeze(1) - AZ).pow(2).sum()
        else:
            # L = D - A
            AZ = torch.sparse.mm(A, Z)
            smooth = ((deg.unsqueeze(1) * Z - AZ) ** 2).sum()
        loss = self.lam * smooth / (num_nodes + 1e-8)
        return loss, {"graph_smooth": float(loss.detach().cpu())}