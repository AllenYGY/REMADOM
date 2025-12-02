from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import anndata as ad  # type: ignore
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover - optional dependency handling
    ad = None  # type: ignore
    sp = None  # type: ignore

from .registries import Registry


def _mk_get(mk: Any, key: str, default: Any = None) -> Any:
    if isinstance(mk, dict):
        return mk.get(key, default)
    return getattr(mk, key, default)


def _get_feature_names(adata, mk: Any, modality: str) -> List[str]:
    if modality == "adt":
        uns_key = _mk_get(mk, "uns_key", "adt_names")
        if uns_key and uns_key in adata.uns:
            names = adata.uns[uns_key]
            return list(map(str, list(names)))
    var_key = _mk_get(mk, "var_key", None)
    if var_key and var_key in adata.var.columns:
        return list(map(str, list(adata.var[var_key].values)))
    return list(map(str, list(adata.var_names)))


def _get_matrix_handle(adata, mk: Any):
    obsm_key = _mk_get(mk, "obsm", None)
    if obsm_key:
        if obsm_key not in adata.obsm:
            raise KeyError(f"obsm key {obsm_key} not found")
        return adata.obsm[obsm_key]
    X_key = _mk_get(mk, "X", "X")
    if X_key == "X":
        return adata.X
    if X_key not in adata.layers:
        raise KeyError(f"layer {X_key} not found")
    return adata.layers[X_key]

class AnnDataDataset(Dataset):
    def __init__(
        self,
        adata,
        keys: Dict[str, Any],
        batch_key: str = "batch",
        registry: Optional[Registry] = None,
        lazy: bool = True,
        dense_cache_mods: Optional[List[str]] = None,
    ):
        if adata.is_view:
            adata = adata.copy()
        self.adata = adata
        self.keys = keys
        self.batch_key = batch_key
        self.modalities: List[str] = list(keys.keys())
        self.lazy = lazy
        self._dense_cache_mods = set(dense_cache_mods or [])
        self._batches = adata.obs[batch_key].values if batch_key in adata.obs.columns else np.zeros(adata.n_obs, dtype=int)
        self._datasets = adata.obs["dataset"].values if "dataset" in adata.obs.columns else self._batches

        # For each modality, store handle and column indices aligned to registry
        self._handles: Dict[str, Any] = {}
        self._col_idx: Dict[str, Optional[np.ndarray]] = {}
        self._row_cache: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in self.modalities}
        for mod, mk in keys.items():
            H = _get_matrix_handle(adata, mk)
            names = _get_feature_names(adata, mk, mod)
            if registry is None or registry.get_vocab(mod) is None:
                if hasattr(H, "shape"):
                    vocab_idx = np.arange(H.shape[1], dtype=int)
                else:
                    vocab_idx = None
            else:
                vocab = registry.get_vocab(mod) or []
                name_to_idx = {n: i for i, n in enumerate(names)}
                idx = [name_to_idx[n] for n in vocab if n in name_to_idx]
                if len(idx) == 0 and hasattr(H, "shape"):
                    vocab_idx = np.arange(H.shape[1], dtype=int)
                else:
                    vocab_idx = np.array(idx, dtype=int) if len(idx) > 0 else np.array([], dtype=int)
            self._handles[mod] = H
            self._col_idx[mod] = vocab_idx

            # Optional dense cache upfront (not typical for very large matrices)
            if not lazy:
                # Materialize full matrix slice aligned to vocab_idx
                M = self._extract_full_matrix(H, vocab_idx)
                # Replace handle with materialized array
                self._handles[mod] = M
                self._col_idx[mod] = None

    def _extract_full_matrix(self, H, idx: Optional[np.ndarray]) -> np.ndarray:
        # Convert handle H to numpy array, column-select if idx provided
        if sp is not None and sp.issparse(H):
            M = H.toarray()
        else:
            try:
                M = np.asarray(H)
            except Exception:
                # Some backings may require explicit reading
                M = np.asarray(H[:,:])
        if idx is None:
            return M
        return M[:, idx] if idx.size > 0 else np.zeros((M.shape[0], 0), dtype=M.dtype)

    def __len__(self) -> int:
        return int(self.adata.n_obs)

    def _get_row(self, mod: str, i: int) -> np.ndarray:
        # Check cache
        if mod in self._dense_cache_mods and i in self._row_cache[mod]:
            return self._row_cache[mod][i]
        H = self._handles[mod]
        idx = self._col_idx[mod]
        # Extract row vector lazily
        if sp is not None and sp.issparse(H):
            row = H[i]
            row = row.toarray().ravel()
        else:
            try:
                row = np.asarray(H[i]).ravel()
            except Exception:
                # For h5backed, slicing returns numpy array already
                row = np.array(H[i]).ravel()
        if idx is not None:
            row = row[idx] if idx.size > 0 else np.zeros((0,), dtype=row.dtype)
        if mod in self._dense_cache_mods:
            self._row_cache[mod][i] = row
        return row

    def __getitem__(self, i: int):
        item: Dict[str, Any] = {}
        for mod in self.modalities:
            # if lazy False and handle is full numpy, _get_row still works
            x = self._get_row(mod, i)
            xt = torch.tensor(x, dtype=torch.float32)
            item[f"x_{mod}"] = xt
            if f"has_{mod}" in self.adata.obs.columns:
                flag = bool(self.adata.obs[f"has_{mod}"].iloc[i])
            else:
                flag = xt.numel() > 0
            item[f"has_{mod}"] = torch.tensor(flag, dtype=torch.bool)
        item["batch_labels"] = torch.tensor(int(self._batches[i]), dtype=torch.long)
        item["dataset_labels"] = torch.tensor(int(self._datasets[i]), dtype=torch.long)
        item["indices"] = torch.tensor(i, dtype=torch.long)
        return item

def batch_collate(items: List[Dict[str, Any]]):
    from ..typing import Batch

    def stack_field(prefix: str, m: str):
        k = f"{prefix}_{m}"
        vals = [it[k] for it in items if k in it]
        return torch.stack(vals, 0) if len(vals) > 0 else None
    x_rna = stack_field("x", "rna")
    x_atac = stack_field("x", "atac")
    x_adt = stack_field("x", "adt")
    has_rna = stack_field("has", "rna")
    has_atac = stack_field("has", "atac")
    has_adt = stack_field("has", "adt")
    lib = x_rna.sum(1) if x_rna is not None else None
    batches = torch.stack([it["batch_labels"] for it in items], 0) if "batch_labels" in items[0] else None
    datasets = torch.stack([it["dataset_labels"] for it in items], 0) if "dataset_labels" in items[0] else None
    indices = torch.stack([it["indices"] for it in items], 0) if "indices" in items[0] else None
    return Batch(
        x_rna=x_rna, x_atac=x_atac, x_adt=x_adt,
        has_rna=has_rna, has_atac=has_atac, has_adt=has_adt,
        libsize_rna=lib, batch_labels=batches, dataset_labels=datasets,
        indices=indices
    )
