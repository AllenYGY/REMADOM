from __future__ import annotations
from typing import Optional
try:
    import anndata as ad  # type: ignore
except Exception:
    ad = None  # type: ignore

def read_h5ad_backed(path: str, mode: str = "r"):
    """
    Open AnnData in backed mode (X only). For broader backing (layers/obsm),
    prefer Zarr-backed AnnData if available.
    """
    if ad is None:
        raise RuntimeError("anndata not available")
    return ad.read_h5ad(path, backed=mode)