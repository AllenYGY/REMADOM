from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor
import numpy as np

def trustworthiness(X: Tensor, X_emb: Tensor, n_neighbors: int = 30) -> float:
    with torch.no_grad():
        D = torch.cdist(X, X)
        De = torch.cdist(X_emb, X_emb)
        nn = torch.topk(-D, k=min(n_neighbors + 1, X.shape[0]), dim=1).indices[:, 1:]
        nne = torch.topk(-De, k=min(n_neighbors + 1, X_emb.shape[0]), dim=1).indices[:, 1:]
        n = X.shape[0]
        ranks = torch.argsort(D, dim=1)
        score = 0.0
        for i in range(n):
            set_true = set(nn[i].tolist())
            for j in nne[i].tolist():
                if j not in set_true:
                    r = (ranks[i] == j).nonzero(as_tuple=False).item()
                    score += max(0, r - n_neighbors)
        denom = n * n_neighbors * (2 * n - 3 * n_neighbors - 1) + 1e-8
        t = 1.0 - (2.0 / denom) * score
        return float(t)

def foscttm(Z_a: Tensor, Z_b: Tensor, matches: Tensor) -> float:
    with torch.no_grad():
        D = torch.cdist(Z_a, Z_b)
        n = Z_a.shape[0]
        ranks = torch.argsort(D, dim=1)
        counts = []
        for i in range(n):
            true_j = int(matches[i].item())
            order = (ranks[i] == true_j).nonzero(as_tuple=False)
            if order.numel() == 0:
                counts.append(D.shape[1])
            else:
                counts.append(int(order[0, 0].item()))
        return float(np.mean(np.array(counts) / D.shape[1]))

def coupling_entropy(P: Tensor, eps: float = 1e-12) -> float:
    with torch.no_grad():
        p = P / (P.sum() + eps)
        h = -(p * (p + eps).log()).sum()
        return float(h.cpu())

def batch_classifier_auc(Z: Tensor, batches: Tensor, hidden: int = 64, epochs: int = 5) -> float:
    # simple logistic regression baseline; AUC would need sklearn; here we return train accuracy as proxy
    n, d = Z.shape
    k = int(torch.max(batches).item() + 1)
    W = torch.zeros(d, k, device=Z.device, dtype=Z.dtype, requires_grad=True)
    opt = torch.optim.SGD([W], lr=0.1)
    for _ in range(epochs):
        logits = Z @ W
        loss = torch.nn.functional.cross_entropy(logits, batches)
        opt.zero_grad()
        loss.backward()
        opt.step()
    preds = torch.argmax(Z @ W, dim=1)
    acc = (preds == batches).float().mean().item()
    return float(acc)