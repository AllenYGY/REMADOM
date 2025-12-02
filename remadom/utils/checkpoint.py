from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


def load_checkpoint(path: str) -> Dict[str, object]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(p, map_location="cpu")


def resume_trainer(trainer, checkpoint_path: str) -> None:
    state = load_checkpoint(checkpoint_path)
    trainer.load_checkpoint(state)
