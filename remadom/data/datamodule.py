from __future__ import annotations

from typing import Optional, Tuple

from torch.utils.data import DataLoader

from ..config.schema import ExperimentConfig
from .loaders import build_dataloaders


class CompositeDataModule:
    """
    Lightweight container mirroring Lightning-style data modules.
    """

    def __init__(self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        try:
            self.steps_per_epoch = len(train_loader)
        except Exception:
            self.steps_per_epoch = None

    @classmethod
    def from_config(
        cls,
        cfg: ExperimentConfig,
        *,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
    ) -> "CompositeDataModule":
        train_loader, val_loader, _ = build_dataloaders(
            cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return cls(train_loader, val_loader)
