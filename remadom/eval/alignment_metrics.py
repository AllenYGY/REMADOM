from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

def compute_alignment_metrics(z: Tensor, groups: Tensor) -> Dict[str, float]:
    """
    Compute lightweight alignment diagnostics:
    - within-group dispersion (mean squared distance to group mean)
    - between-group distance (mean squared distance between group centroids)
    - centroid variance (variance of group means)
    - silhouette proxy: (nearest-other-centroid distance - own-centroid distance) / max(...)
    """
    if z.numel() == 0 or groups.numel() == 0:
        return {}
    device = z.device
    groups = groups.to(device)
    unique = torch.unique(groups)
    if unique.numel() == 0:
        return {}

    centroids = []
    within = []
    for g in unique:
        mask = groups == g
        if mask.sum() == 0:
            continue
        vals = z[mask]
        mean = vals.mean(dim=0, keepdim=True)
        centroids.append(mean)
        dispersion = torch.mean((vals - mean).pow(2).sum(dim=1))
        within.append(dispersion)
    if not centroids:
        return {}
    centroids_tensor = torch.cat(centroids, dim=0)
    centroid_var = torch.var(centroids_tensor, dim=0).mean()

    if centroids_tensor.shape[0] > 1:
        dists = torch.cdist(centroids_tensor, centroids_tensor, p=2)
        # take upper triangle without diagonal
        triu = torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)
        between = (dists[triu[0], triu[1]] ** 2).mean()
    else:
        between = torch.tensor(0.0, device=device)

    within_mean = torch.stack(within).mean() if within else torch.tensor(0.0, device=device)

    # Silhouette-like proxy using centroids (avoids full pairwise)
    sil_scores = []
    for g in unique:
        mask = groups == g
        if mask.sum() == 0:
            continue
        vals = z[mask]
        own_centroid = centroids_tensor[unique == g][0]
        a = torch.norm(vals - own_centroid, dim=1).mean()
        # nearest other centroid
        others = centroids_tensor[unique != g]
        if others.numel() == 0:
            continue
        dists_other = torch.cdist(vals, others).mean(dim=0)
        b = dists_other.min()
        sil = (b - a) / (torch.max(a, b) + torch.finfo(z.dtype).eps)
        sil_scores.append(sil)
    silhouette = torch.stack(sil_scores).mean() if sil_scores else torch.tensor(0.0, device=device)

    return {
        "within_dispersion": float(within_mean.detach().cpu()),
        "between_centroid_dist": float(between.detach().cpu()),
        "centroid_variance": float(centroid_var.detach().cpu()),
        "silhouette_proxy": float(silhouette.detach().cpu()),
    }


def compute_batch_metrics(z: Tensor, X: Tensor, groups: Tensor, batch_key: str = "batch") -> Dict[str, float]:
    """
    Optional SCIB-like metrics: ilisi, kBET if scib is available.
    """
    if np is None:
        return {}
    from .scib_wrapper import compute_batch_metrics as _batch_metrics  # type: ignore

    try:
        z_np = z.detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        g_np = groups.detach().cpu().numpy()
    except Exception:
        return {}
    return _batch_metrics(X_np, z_np, g_np, batch_key=batch_key)
