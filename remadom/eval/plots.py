from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional
    plt = None  # type: ignore


def latent_umap(z: np.ndarray, groups: Optional[np.ndarray] = None, path: Optional[str] = None, *, title: str = "Latent UMAP") -> Optional[str]:
    try:
        import umap  # type: ignore
    except Exception:
        return None
    if plt is None:
        return None
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="euclidean")
    emb = reducer.fit_transform(z)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=groups, s=6, cmap="tab10" if groups is not None else None, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if groups is not None:
        plt.colorbar(sc, ax=ax, label="group")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
        plt.close(fig)
        return path
    return None


def latent_tsne(z: np.ndarray, groups: Optional[np.ndarray] = None, path: Optional[str] = None, *, title: str = "Latent t-SNE") -> Optional[str]:
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception:
        return None
    if plt is None:
        return None
    emb = TSNE(n_components=2, init="pca", perplexity=30, n_iter=1000).fit_transform(z)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=groups, s=6, cmap="tab10" if groups is not None else None, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    if groups is not None:
        plt.colorbar(sc, ax=ax, label="group")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
        plt.close(fig)
        return path
    return None
