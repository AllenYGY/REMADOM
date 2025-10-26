from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
try:
    import anndata as ad  # type: ignore
except Exception:
    ad = None  # type: ignore
import numpy as np

def align_features(adata, registry: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Minimal adapter: align RNA, ATAC, ADT to registry vocabs if provided.
    Return (X_rna, X_atac, X_adt) as numpy arrays in the registry order (or None).
    Registry expected format:
      { "rna": ["geneA", ...], "atac": ["chr1:...", ...], "adt": ["CD3", ...] }
    """
    if ad is None:
        raise RuntimeError("anndata is required for adapters")

    X_rna = None
    X_atac = None
    X_adt = None

    if "rna" in registry and hasattr(adata, "var") and "gene_ids" in adata.var.columns:
        vocab_reg = registry["rna"]
        vocab_cur = list(adata.var["gene_ids"].values)
        aligned = [vocab_cur.index(g) for g in vocab_reg if g in set(vocab_cur)]
        if len(aligned) > 0:
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            X_rna = np.asarray(X)[:, aligned]

    # Similar logic for atac and adt if available in layer/obsm
    if "atac" in registry and "X_atac" in adata.layers:
        vocab_cur = list(adata.var_names)  # or a var-peaks slot
        vocab_reg = registry["atac"]
        aligned = [vocab_cur.index(p) for p in vocab_reg if p in set(vocab_cur)]
        Xa = adata.layers["X_atac"].toarray() if hasattr(adata.layers["X_atac"], "toarray") else adata.layers["X_atac"]
        X_atac = np.asarray(Xa)[:, aligned] if len(aligned) > 0 else None

    if "adt" in registry and "X_adt" in adata.obsm_keys():
        vocab_cur = list(adata.uns.get("adt_names", []))
        vocab_reg = registry["adt"]
        aligned = [vocab_cur.index(p) for p in vocab_reg if p in set(vocab_cur)]
        X_adt = np.asarray(adata.obsm["X_adt"])[:, aligned] if len(aligned) > 0 else None

    return X_rna, X_atac, X_adt