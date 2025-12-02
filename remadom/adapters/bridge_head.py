from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..align.base import AlignmentHead
from ..bridges.base import BridgeEdges, BridgeProvider
from ..bridges.mnn import MNNBridge
from ..bridges.seeded import SeededBridge
from ..bridges.dictionary import DictionaryBridgeProvider
from ..bridges.linmap import LinearMapBridge
from ..bridges.utils import diagnostics as bridge_diagnostics

def build_bridge_provider(method: str, params: Optional[Dict[str, object]] = None) -> BridgeProvider:
    params = params or {}
    name = method.lower()
    if name == "mnn":
        k = int(params.get("k", 20))
        metric = str(params.get("metric", "euclidean"))
        backend = str(params.get("backend", "faiss"))
        return MNNBridge(k=k, metric=metric, backend=backend)
    if name == "seeded":
        seeds = params.get("seed_pairs")
        if seeds is not None and not isinstance(seeds, (list, tuple)):
            raise TypeError("seed_pairs must be a list of (i, j) tuples")
        radius = int(params.get("radius", 0))
        return SeededBridge(seed_pairs=seeds, radius=radius)
    if name == "dictionary":
        bridge_size = params.get("bridge_size")
        lam = float(params.get("lam", 0.0))
        return DictionaryBridgeProvider(bridge_size=bridge_size, lam=lam)
    raise ValueError(f"Unsupported bridge provider '{method}'")


class BridgeHead(AlignmentHead):
    """
    Alignment head that penalises differences across bridge edges constructed from latent embeddings.
    """

    def __init__(
        self,
        provider: BridgeProvider,
        *,
        weight: float = 1.0,
        group_key: str = "dataset",
        name: str = "bridge",
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
        normalize: bool = False,
        max_edges: Optional[int] = None,
        allowed_groups: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(weight=weight, name=name)
        self.provider = provider
        self.group_key = group_key
        if pairs is not None:
            self._pairs = [(int(a), int(b)) for a, b in pairs]
        else:
            self._pairs = None
        self.normalize = bool(normalize)
        self.max_edges = None if max_edges is None else int(max_edges)
        self.allowed_groups = set(int(g) for g in allowed_groups) if allowed_groups is not None else None

    def set_params(self, **kwargs) -> None:  # pragma: no cover - exercised via Trainer schedule
        super().set_params(**kwargs)
        if "weight" in kwargs:
            self.weight = float(kwargs["weight"])

    def forward(
        self,
        z_bio: Tensor,
        groups: Optional[Tensor] = None,
        graph: Optional[object] = None,
        bridge: Optional[object] = None,
        aux: Optional[Dict[str, object]] = None,
    ) -> Tuple[Tensor, Dict[str, object]]:
        if groups is None:
            return torch.tensor(0.0, device=z_bio.device), {"bridge_edges": 0}
        unique = torch.unique(groups)
        if unique.numel() < 2:
            return torch.tensor(0.0, device=z_bio.device), {"bridge_edges": 0}

        total_loss = torch.tensor(0.0, device=z_bio.device)
        total_edges = 0
        per_pair_logs: List[Dict[str, object]] = []

        # Determine group pairs to evaluate
        if self._pairs:
            candidate_pairs: Iterable[Tuple[int, int]] = [
                (a, b)
                for a, b in self._pairs
                if (groups == a).any() and (groups == b).any()
            ]
        else:
            vals = unique.tolist()
            candidate_pairs = [(int(vals[i]), int(vals[j])) for i in range(len(vals)) for j in range(i + 1, len(vals))]

        for ga, gb in candidate_pairs:
            if self.allowed_groups is not None and (ga not in self.allowed_groups or gb not in self.allowed_groups):
                continue
            mask = (groups == ga) | (groups == gb)
            if not mask.any():
                continue
            subset_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            subset_groups = groups[subset_idx]
            subset_z = z_bio[subset_idx]
            try:
                edges = self.provider.build(subset_z, subset_groups)
            except AssertionError:
                continue
            if edges.src_idx.numel() == 0:
                continue
            global_edges = BridgeEdges(
                src_idx=subset_idx[edges.src_idx],
                dst_idx=subset_idx[edges.dst_idx],
                weight=edges.weight,
            )
            if self.max_edges is not None and global_edges.src_idx.numel() > self.max_edges:
                keep = slice(0, self.max_edges)
                global_edges = BridgeEdges(
                    src_idx=global_edges.src_idx[keep],
                    dst_idx=global_edges.dst_idx[keep],
                    weight=None if global_edges.weight is None else global_edges.weight[keep],
                )
            diff = z_bio[global_edges.src_idx] - z_bio[global_edges.dst_idx]
            if self.normalize:
                pair_loss = diff.pow(2).mean(dim=1)
            else:
                pair_loss = diff.pow(2).sum(dim=1)
            if global_edges.weight is not None:
                w = global_edges.weight.to(diff.device)
                norm = torch.clamp(w.sum(), min=torch.finfo(diff.dtype).eps)
                pair_loss = (pair_loss * w) / norm
            loss_value = pair_loss.mean()
            total_loss = total_loss + loss_value
            total_edges += int(global_edges.src_idx.numel())
            per_pair_logs.append(
                {
                    "pair": (int(ga), int(gb)),
                    "edges": int(global_edges.src_idx.numel()),
                    **bridge_diagnostics(global_edges, z_bio.shape[0]),
                }
            )

        if total_edges == 0:
            return torch.tensor(0.0, device=z_bio.device), {"bridge_edges": 0}

        total_loss = total_loss / max(1, len(per_pair_logs))
        log_payload: Dict[str, object] = {
            "bridge_edges": total_edges,
            "bridge_pairs": per_pair_logs,
        }
        return self.weight * total_loss, log_payload
