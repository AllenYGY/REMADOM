from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    """
    Dense NPY/NPZ-backed modality dataset.
    Expects a dict mapping modality name to path of .npy/.npz, and a 'batches' path.
    Example:
      NpyDataset({"rna": "rna.npy", "batches": "batches.npy"}, modalities=["rna"])
    """
    def __init__(self, paths: Dict[str, str], modalities: Optional[List[str]] = None, dtype: torch.dtype = torch.float32):
        assert "batches" in paths, "paths must include 'batches' npy path"
        self.modalities = modalities or [k for k in paths.keys() if k != "batches"]
        self.paths = paths
        self.dtype = dtype
        # Load arrays lazily via mmap where possible
        self._arrays: Dict[str, np.ndarray] = {}
        n_obs = None
        for mod in self.modalities:
            path = paths[mod]
            if path.endswith(".npz"):
                with np.load(path) as f:
                    X = f["X"]
            else:
                X = np.load(path, mmap_mode="r")
            self._arrays[mod] = X
            n_obs = int(X.shape[0]) if n_obs is None else n_obs
            if int(X.shape[0]) != n_obs:
                raise ValueError(f"Modality {mod} n_obs mismatch: {X.shape[0]} vs {n_obs}")
        self.batches = np.load(paths["batches"], mmap_mode="r")
        if n_obs is None:
            raise ValueError("No modalities provided")
        if int(self.batches.shape[0]) != n_obs:
            raise ValueError("batches length mismatch with modalities")
        self.n = n_obs

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {}
        for mod in self.modalities:
            x = np.asarray(self._arrays[mod][i]).ravel()
            t = torch.tensor(x, dtype=self.dtype)
            item[f"x_{mod}"] = t
            item[f"has_{mod}"] = torch.tensor(t.numel() > 0, dtype=torch.bool)
        item["batch_labels"] = torch.tensor(int(self.batches[i]), dtype=torch.long)
        return item