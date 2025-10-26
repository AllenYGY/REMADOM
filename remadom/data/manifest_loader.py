from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import json
from torch.utils.data import DataLoader
from .npy_dataset import NpyDataset
from .csr_dataset import CSRDenseDataset
from .zarr_dataset import ZarrDataset
from .composite_dataset import CompositeDataset, composite_collate
try:
    import anndata as ad  # type: ignore
except Exception:  # pragma: no cover
    ad = None

def load_from_manifest(manifest_path: str, batch_size: int = 256, num_workers: int = 0, pin_memory: bool = False) -> Tuple[DataLoader, Optional[DataLoader]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)
    dl_train = _build_split("train", man, batch_size, num_workers, pin_memory)
    dl_valid = _build_split("valid", man, batch_size, num_workers, pin_memory) if "valid" in man.get("splits", {}) else None
    return dl_train, dl_valid

def _build_split(name: str, man: Dict[str, Any], batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    split = man["splits"][name]
    mods_meta = split["modalities"]
    batches_path = split.get("batches")
    batch_key = split.get("batch_key", "batch")
    reg = None
    # Optional: load registry if needed elsewhere
    mod_datasets = {}
    for mod, meta in mods_meta.items():
        t = meta["type"]
        if t == "csr":
            ds = CSRDenseDataset({mod: {"shards": meta["shards"], "shape": tuple(meta["shape"])}}, batches_path=batches_path)
        elif t == "zarr":
            if ad is None:
                raise RuntimeError("anndata is required for zarr-backed entries")
            A = ad.read_zarr(meta["path"])
            ds = ZarrDataset(A, keys=meta["keys"], batch_key=batch_key, registry=reg)
        else:
            ds = NpyDataset({mod: meta["path"], "batches": batches_path}, modalities=[mod])
        mod_datasets[mod] = ds
    ds_comp = CompositeDataset(mod_datasets)
    dl = DataLoader(ds_comp, batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers, pin_memory=pin_memory, collate_fn=composite_collate)
    return dl