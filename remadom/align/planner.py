from __future__ import annotations
from typing import Iterator, List, Tuple, Optional, Dict

class PairPlanner:
    """
    Iterates modality pairs based on a scheme and optional weights.
    """
    def __init__(self, scheme: str = "all_pairs", weights: Optional[Dict[Tuple[str, str], float]] = None):
        self.scheme = scheme
        self.weights = weights or {}

    def iter_pairs(self, modalities: List[str]) -> Iterator[Tuple[str, str, float]]:
        if self.scheme == "all_pairs":
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    a, b = modalities[i], modalities[j]
                    yield a, b, float(self.weights.get((a, b), 1.0))
        elif self.scheme == "star" and modalities:
            center = modalities[0]
            for m in modalities[1:]:
                yield center, m, float(self.weights.get((center, m), 1.0))
        else:
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    yield modalities[i], modalities[j], 1.0