from __future__ import annotations
from typing import Optional
from torch.utils.data import DataLoader
from .torch_dataset import AnnDataDataset, batch_collate
from .registries import Registry
from ..config.schema import DataSource
from .loaders import load_anndata, build_registry_from_adata

def make_torch_loaders(
    train_src: DataSource,
    valid_src: Optional[DataSource] = None,
    registry: Optional[Registry] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    # Build/load registry from train
    adata_train = load_anndata(train_src.path)
    reg = registry or build_registry_from_adata(adata_train, train_src.keys)
    ds_train = AnnDataDataset(adata_train, keys=train_src.keys, batch_key=train_src.batch_key, registry=reg)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=batch_collate)

    dl_valid = None
    if valid_src is not None and valid_src.path:
        adata_valid = load_anndata(valid_src.path)
        ds_valid = AnnDataDataset(adata_valid, keys=valid_src.keys, batch_key=valid_src.batch_key, registry=reg)
        dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=batch_collate)

    return dl_train, dl_valid, reg