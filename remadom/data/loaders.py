from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config.schema import DataSource, ExperimentConfig, ModalityKeyConfig
from ..typing import Batch
from .registries import Registry
from .torch_dataset import AnnDataDataset, batch_collate

try:
    import anndata as ad  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    ad = None
    _ANNDATA_IMPORT_ERROR = exc
else:
    _ANNDATA_IMPORT_ERROR = None


def ensure_anndata_available() -> None:
    if ad is None:
        raise RuntimeError("anndata is not available; install anndata to use data loaders") from _ANNDATA_IMPORT_ERROR


def load_anndata(path: str):
    ensure_anndata_available()
    return ad.read_h5ad(path)  # type: ignore[union-attr]


def _as_modality_key(cfg: ModalityKeyConfig | Dict[str, object]) -> ModalityKeyConfig:
    if isinstance(cfg, ModalityKeyConfig):
        return cfg
    return ModalityKeyConfig(**cfg)


def _extract_matrix(adata, mk: ModalityKeyConfig):
    if mk.obsm:
        if mk.obsm not in adata.obsm:
            raise KeyError(f"obsm key '{mk.obsm}' not found in AnnData")
        return adata.obsm[mk.obsm]
    X_key = mk.X or "X"
    if X_key == "X":
        return adata.X
    if X_key not in adata.layers:
        raise KeyError(f"layer '{X_key}' not found in AnnData")
    return adata.layers[X_key]


def _feature_names(adata, mk: ModalityKeyConfig, modality: str) -> list[str]:
    if mk.obsm:
        matrix = adata.obsm[mk.obsm]
        if modality == "adt":
            uns_key = mk.uns_key or "adt_names"
            if uns_key in adata.uns:
                return list(map(str, list(adata.uns[uns_key])))
        return [f"{modality}_feature_{i}" for i in range(matrix.shape[1])]
    if modality == "adt":
        uns_key = mk.uns_key or "adt_names"
        if uns_key in adata.uns:
            return list(map(str, list(adata.uns[uns_key])))
    if mk.var_key and mk.var_key in adata.var.columns:
        return list(map(str, list(adata.var[mk.var_key].values)))
    return list(map(str, list(adata.var_names)))


def build_registry_from_adata(adata, keys: Dict[str, ModalityKeyConfig | Dict[str, object]]) -> Registry:
    reg = Registry(name="auto")
    for mod, mk_cfg in keys.items():
        mk = _as_modality_key(mk_cfg)
        names = _feature_names(adata, mk, mod)
        reg.set_vocab(mod, names)
    return reg


def align_to_registry(adata, reg: Registry, keys: Dict[str, ModalityKeyConfig | Dict[str, object]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for mod, mk_cfg in keys.items():
        mk = _as_modality_key(mk_cfg)
        matrix = _extract_matrix(adata, mk)
        names = _feature_names(adata, mk, mod)
        vocab = reg.get_vocab(mod)
        if vocab is None:
            arr = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        else:
            name_to_idx = {n: i for i, n in enumerate(names)}
            idx = [name_to_idx[n] for n in vocab if n in name_to_idx]
            if hasattr(matrix, "toarray"):
                arr = matrix.toarray()[:, idx]
            else:
                arr = np.asarray(matrix)[:, idx]
        out[mod] = arr
    return out


def dataloader_from_anndata(
    adata,
    *,
    keys: Dict[str, ModalityKeyConfig | Dict[str, object]],
    batch_key: str = "batch",
    registry: Optional[Registry] = None,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Iterable[Batch]:
    dataset = AnnDataDataset(adata, keys=keys, batch_key=batch_key, registry=registry)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=batch_collate,
    )


def dataloader_from_source(
    source: DataSource,
    *,
    registry: Optional[Registry] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[Iterable[Batch], Registry]:
    adata = load_anndata(source.path)
    reg = registry or build_registry_from_adata(adata, source.keys)
    loader = dataloader_from_anndata(
        adata,
        keys=source.keys,
        batch_key=source.batch_key,
        registry=reg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, reg


def build_dataloaders(
    cfg: ExperimentConfig,
    *,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]], Registry]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = batch_size or cfg.optim.batch_size
    pin = bool(pin_memory if pin_memory is not None else device.type == "cuda")
    train_loader, registry = dataloader_from_source(
        cfg.data.source,
        registry=None,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = None
    if cfg.data.valid is not None:
        val_loader, _ = dataloader_from_source(
            cfg.data.valid,
            registry=registry,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        )
    return train_loader, val_loader, registry


def dataloader(
    adata,
    cfg,
    *,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    registry: Optional[Registry] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Iterable[Batch]:
    if hasattr(cfg, "data") and hasattr(cfg.data, "source"):
        keys = cfg.data.source.keys
        batch_key = cfg.data.source.batch_key
    elif hasattr(cfg, "data") and hasattr(cfg.data, "keys"):
        keys = cfg.data.keys  # type: ignore[attr-defined]
        batch_key = getattr(cfg.data, "batch_key", "batch")  # type: ignore[attr-defined]
    else:
        raise AttributeError("Configuration is missing data source information")
    bs = batch_size or getattr(cfg.optim, "batch_size", 256)
    return dataloader_from_anndata(
        adata,
        keys=keys,
        batch_key=batch_key,
        registry=registry,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
