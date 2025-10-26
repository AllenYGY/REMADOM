from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn.functional as F

class FixedMixtureMapper:
    """
    Simple prototype-based mapper over reference embeddings and batch/label anchors.
    fit(Z_ref, y_ref): learn class means; map(Z_q) returns argmax probs.
    """
    def __init__(self):
        self.mu: Optional[torch.Tensor] = None  # (C, D)
        self.classes: Optional[torch.Tensor] = None

    def fit(self, Z: torch.Tensor, y: torch.Tensor) -> "FixedMixtureMapper":
        classes = torch.unique(y)
        mus = []
        for c in classes:
            mus.append(Z[y == c].mean(0, keepdim=True))
        self.mu = torch.cat(mus, 0)  # (C, D)
        self.classes = classes
        return self

    def map(self, Z: torch.Tensor, return_probs: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mu is None or self.classes is None:
            raise RuntimeError("Call fit() first")
        # Cosine similarity to means
        Zn = F.normalize(Z, dim=1)
        Mun = F.normalize(self.mu, dim=1)
        sims = Zn @ Mun.T  # (N, C)
        probs = F.softmax(sims, dim=1)
        idx = probs.argmax(dim=1)
        preds = self.classes[idx]
        return (preds, probs) if return_probs else (preds, None)