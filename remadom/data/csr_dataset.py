from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None

def _parse_span_from_name(path: str) -> Tuple[int, int]:
    base = os.path.basename(path)
    m = re.search(r"_part(\d+)_(\d+)\.npz$", base)
    if not m:
        raise ValueError(f"Cannot parse span from shard name: {base}")
    return int(m.group(1)), int(m.group(2))

class CSRShards:
    """
    Lazy loader with LRU cache for CSR shard matrices.
    """
    def __init__(self, shard_paths: List[str], row_spans: List[Tuple[int, int]], cache_capacity: int = 2):
        if sp is None:
            raise RuntimeError("scipy.sparse is required for CSR shards")
        if len(shard_paths) != len(row_spans):
            raise ValueError("shard_paths and row_spans lengths mismatch")
        self.paths = shard_paths
        self.spans = row_spans
        self.cache_capacity = max(1, int(cache_capacity))
        self._cache: "dict[int, sp.csr_matrix]" = {}
        self._lru: List[int] = []

    def _touch(self, k: int) -> None:
        if k in self._lru:
            self._lru.remove(k)
        self._lru.append(k)
        if len(self._lru) > self.cache_capacity:
            evict = self._lru.pop(0)
            self._cache.pop(evict, None)

    def _find_shard(self, i: int) -> Tuple[int, int]:
        for k, (s, e) in enumerate(self.spans):
            if s <= i < e:
                return k, i - s
        raise IndexError(i)

    def _load(self, k: int):
        if k in self._cache:
            self._touch(k)
            return self._cache[k]
        M = sp.load_npz(self.paths[k])
        self._cache[k] = M
        self._touch(k)
        return M

    def get_row(self, i: int) -> np.ndarray:
        k, local = self._find_shard(i)
        M = self._load(k)
        return M.getrow(local).toarray().ravel()

class CSRDenseDataset(Dataset):
    """
    Serve dense rows from CSR shards for one or more modalities.
    modalities meta example:
      {
        "atac": {"shards": ["atac_part0_5000.npz", ...], "shape": [N, P]}
      }
    """
    def __init__(self, modalities: Dict[str, Dict[str, Any]], batches_path: str, cache_capacity: int = 2, dtype: torch.dtype = torch.float32):
        if sp is None:
            raise RuntimeError("scipy.sparse is required for CSR dataset")
        self.modalities = list(modalities.keys())
        shapes = {mod: tuple(meta["shape"]) for mod, meta in modalities.items()}
        n_set = {int(s[0]) for s in shapes.values()}
        if len(n_set) != 1:
            raise ValueError(f"All modalities must share same n_obs, got {shapes}")
        self.n = n_set.pop()
        self.dtype = dtype
        self.shards: Dict[str, CSRShards] = {}
        for mod, meta in modalities.items():
            spans: List[Tuple[int, int]] = []
            for p in meta["shards"]:
                spans.append(_parse_span_from_name(p))
            spans.sort(key=lambda x: x[0])
            self.shards[mod] = CSRShards(meta["shards"], spans, cache_capacity=cache_capacity)
        self.batches = np.load(batches_path, mmap_mode="r")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {}
        for mod in self.modalities:
            row = self.shards[mod].get_row(int(i))
            t = torch.tensor(row, dtype=self.dtype)
            item[f"x_{mod}"] = t
            item[f"has_{mod}"] = torch.tensor(t.numel() > 0, dtype=torch.bool)
        item["batch_labels"] = torch.tensor(int(self.batches[i]), dtype=torch.long)
        return item