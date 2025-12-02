from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class Batch:
    x_rna: Optional[torch.Tensor] = None
    x_atac: Optional[torch.Tensor] = None
    x_adt: Optional[torch.Tensor] = None
    has_rna: Optional[torch.Tensor] = None
    has_atac: Optional[torch.Tensor] = None
    has_adt: Optional[torch.Tensor] = None
    libsize_rna: Optional[torch.Tensor] = None
    batch_labels: Optional[torch.Tensor] = None
    dataset_labels: Optional[torch.Tensor] = None
    time: Optional[torch.Tensor] = None
    coords: Optional[torch.Tensor] = None
    modality_masks: Optional[Dict[str, torch.Tensor]] = None
    meta: Optional[Dict[str, Any]] = None
    indices: Optional[torch.Tensor] = None
