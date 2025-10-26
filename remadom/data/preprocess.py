from __future__ import annotations
from typing import Any, Dict
try:
    from anndata import AnnData  # type: ignore
except Exception:
    AnnData = object  # type: ignore

def preprocess_rna(adata: AnnData, cfg: Dict[str, Any]) -> AnnData:
    # Placeholder: user can plug in scanpy pipelines; here we assume X is already usable
    return adata

def preprocess_atac(adata: AnnData, cfg: Dict[str, Any]) -> AnnData:
    return adata

def preprocess_adt(adata: AnnData, cfg: Dict[str, Any]) -> AnnData:
    return adata