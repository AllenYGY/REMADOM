from __future__ import annotations
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset

class CompositeDataset(Dataset):
    """
    Merge multiple modality-specific datasets by index into one item dict.
    Assumes equal length and shared row order.
    """
    def __init__(self, datasets: Dict[str, Dataset], modality_order: Optional[List[str]] = None):
        if not datasets:
            raise ValueError("Provide at least one dataset")
        self.datasets = datasets
        self.modalities = modality_order or list(datasets.keys())
        lens = [len(ds) for ds in datasets.values()]
        if len(set(lens)) != 1:
            raise ValueError(f"Datasets must share length, got {lens}")
        self.n = lens[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for mod in self.modalities:
            it = self.datasets[mod][i]
            for k, v in it.items():
                merged[k] = v
        return merged

def composite_collate(items: List[Dict[str, Any]]):
    # Minimal collate producing a standard Batch-like dict.
    # You can replace with a dataclass Batch if your core expects it.
    out: Dict[str, Optional[torch.Tensor]] = {}
    keys = set().union(*[it.keys() for it in items])
    for k in keys:
        vals = [it[k] for it in items if k in it and it[k] is not None]
        if len(vals) == 0:
            out[k] = None
            continue
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, 0)
        else:
            # Non-tensor metadata not stacked here
            out[k] = None
    return out