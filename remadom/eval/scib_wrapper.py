from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
try:
    import anndata as ad  # type: ignore
    import scib  # type: ignore
except Exception:
    ad = None  # type: ignore
    scib = None  # type: ignore

def adata_from_numpy(X: np.ndarray, Z: np.ndarray, batches: np.ndarray, batch_key: str = "batch") -> Optional["ad.AnnData"]:  # type: ignore
    if ad is None:
        return None
    A = ad.AnnData(X)
    A.obs[batch_key] = batches.astype(str)
    A.obsm["X_emb"] = Z
    return A

def compute_ilisi(adata, batch_key: str = "batch") -> Optional[float]:
    if scib is None:
        return None
    try:
        return float(scib.me.ilisi_graph(adata, batch_key=batch_key))  # type: ignore
    except Exception:
        return None

def compute_kbet(adata, batch_key: str = "batch") -> Optional[float]:
    if scib is None:
        return None
    try:
        return float(scib.me.kBET(adata, batch_key=batch_key))  # type: ignore
    except Exception:
        return None


def compute_batch_metrics(X: np.ndarray, Z: np.ndarray, batches: np.ndarray, batch_key: str = "batch") -> Dict[str, float]:
    A = adata_from_numpy(X, Z, batches, batch_key=batch_key)
    if A is None:
        return {}
    out: Dict[str, float] = {}
    ilisi = compute_ilisi(A, batch_key=batch_key)
    if ilisi is not None:
        out["ilisi"] = ilisi
    kbet = compute_kbet(A, batch_key=batch_key)
    if kbet is not None:
        out["kbet"] = kbet
    return out
