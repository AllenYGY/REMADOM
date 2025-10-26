from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import numpy as np
try:
    import anndata as ad  # type: ignore
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    ad = None
    sp = None

def export_aligned_arrays_blockwise(
    adata_path: str,
    keys: Dict[str, Dict[str, Any]],
    registry: Any,
    out_dir: str,
    prefix: str,
    batch_key: str,
    fmt: str = "npy",
    chunk_size: int = 8192,
    mods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Export aligned arrays for selected modalities with registry-aligned columns.
    Returns a dict with per-modality metadata and 'batches' path.
    NOTE: This is a stub — implement alignment to registry vocabularies in your codebase.
    """
    if ad is None:
        raise RuntimeError("anndata required for export")
    os.makedirs(out_dir, exist_ok=True)
    A = ad.read_h5ad(adata_path) if adata_path.endswith(".h5ad") else ad.read_zarr(adata_path)
    result: Dict[str, Any] = {}
    # Batches
    batches_path = os.path.join(out_dir, f"{prefix}_batches.npy")
    np.save(batches_path, A.obs[batch_key].values if batch_key in A.obs.columns else np.zeros(A.n_obs, dtype=int))
    result["batches"] = batches_path
    # For simplicity, we export dense .npy unless fmt=="csr"
    sel_mods = mods or list(keys.keys())
    for mod in sel_mods:
        mk = keys[mod]
        X = _select_matrix(A, mk)
        # TODO: align columns using registry vocab if provided
        if fmt == "csr":
            if sp is None:
                raise RuntimeError("scipy.sparse required for csr export")
            # naive single-shard export in stub
            csr = sp.csr_matrix(X) if not sp.isspmatrix_csr(X) else X
            path = os.path.join(out_dir, f"{prefix}_{mod}_part0_{A.n_obs}.npz")
            sp.save_npz(path, csr)
            result[mod] = {"shards": [path], "shape": [int(A.n_obs), int(X.shape[1])]}
        else:
            path = os.path.join(out_dir, f"{prefix}_{mod}.npy")
            np.save(path, np.asarray(X))
            result[mod] = path
    return result

def _select_matrix(adata, mk: Dict[str, Any]):
    obsm_key = mk.get("obsm")
    if obsm_key:
        return adata.obsm[obsm_key]
    X_key = mk.get("X", "X")
    if X_key == "X":
        return adata.X
    return adata.layers[X_key]