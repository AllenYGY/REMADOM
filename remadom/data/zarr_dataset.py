from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import anndata as ad  # type: ignore
except Exception:  # pragma: no cover
    ad = None

class ZarrDataset(Dataset):
    def __init__(self, adata, keys: Dict[str, Dict[str, Any]], batch_key: str = "batch", registry: Optional[Any] = None, dtype: torch.dtype = torch.float32):
        if ad is None:
            raise RuntimeError("anndata is required for ZarrDataset")
        if adata.is_view:
            adata = adata.copy()
        self.adata = adata
        self.keys = keys
        self.modalities = list(keys.keys())
        self.batch_key = batch_key
        self.dtype = dtype
        self._arrays: Dict[str, Any] = {}
        self._col_idx: Dict[str, Optional[np.ndarray]] = {}
        # Prepare arrays and optional column index alignment
        for mod, mk in keys.items():
            arr = self._get_array(adata, mk)
            self._arrays[mod] = arr
            self._col_idx[mod] = self._build_col_index(adata, mk, registry, mod)
        self._batches = adata.obs[batch_key].values if batch_key in adata.obs.columns else np.zeros(adata.n_obs, dtype=int)
        self.n = int(adata.n_obs)

    def _get_array(self, adata, mk: Dict[str, Any]):
        obsm_key = mk.get("obsm")
        if obsm_key:
            return adata.obsm[obsm_key]
        X_key = mk.get("X", "X")
        if X_key == "X":
            return adata.X
        return adata.layers[X_key]

    def _build_col_index(self, adata, mk: Dict[str, Any], registry: Optional[Any], modality: str) -> Optional[np.ndarray]:
        if registry is None:
            return None
        vocab = getattr(registry, "get_vocab", lambda m: None)(modality)
        if vocab is None:
            return None
        # Feature names
        names = None
        if modality == "adt":
            names = adata.uns.get(mk.get("uns_key", "adt_names"), None)
            if names is None:
                return None
            names = list(map(str, list(names)))
        else:
            var_key = mk.get("var_key")
            if var_key and var_key in adata.var.columns:
                names = list(map(str, list(adata.var[var_key].values)))
            else:
                names = list(map(str, list(adata.var_names)))
        name_to_idx = {n: i for i, n in enumerate(names)}
        idx = [name_to_idx[n] for n in vocab if n in name_to_idx]
        return np.array(idx, dtype=int) if len(idx) > 0 else np.array([], dtype=int)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {}
        for mod in self.modalities:
            A = self._arrays[mod]
            cols = self._col_idx[mod]
            row = np.asarray(A[i]).ravel()
            if cols is not None and cols.size > 0:
                row = row[cols]
            t = torch.tensor(row, dtype=self.dtype)
            item[f"x_{mod}"] = t
            item[f"has_{mod}"] = torch.tensor(t.numel() > 0, dtype=torch.bool)
        item["batch_labels"] = torch.tensor(int(self._batches[i]), dtype=torch.long)
        return item