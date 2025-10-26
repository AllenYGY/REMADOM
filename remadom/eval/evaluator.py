from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
import torch
from torch import Tensor
from .metrics import trustworthiness, foscttm, batch_classifier_auc, coupling_entropy
from .scib_wrapper import adata_from_numpy, compute_ilisi, compute_kbet

class EvalReport:
    def __init__(self, metrics: Dict[str, Any], diagnostics: Dict[str, Any]):
        self.metrics = metrics
        self.diagnostics = diagnostics
    def to_json(self) -> str:
        return json.dumps({"metrics": self.metrics, "diagnostics": self.diagnostics}, indent=2)

class Evaluator:
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names

    def compute_metrics(self, Z: Tensor, batches: Optional[Tensor] = None, aux: Dict[str, Any] | None = None) -> EvalReport:
        metrics: Dict[str, Any] = {}
        # Trustworthiness
        if "trustworthiness" in self.metric_names and aux is not None and "X" in aux:
            metrics["trustworthiness"] = trustworthiness(aux["X"], Z, n_neighbors=aux.get("n_neighbors", 30))
        # Batch classifier proxy
        if "batch_cls_acc" in self.metric_names and batches is not None:
            metrics["batch_cls_acc"] = batch_classifier_auc(Z, batches)
        # scIB ilisi/kbet if provided arrays
        if batches is not None and aux is not None and "X_np" in aux and "Z_np" in aux:
            A = adata_from_numpy(aux["X_np"], aux["Z_np"], batches.detach().cpu().numpy(), batch_key=aux.get("batch_key", "batch"))
            if A is not None:
                if "ilisi" in self.metric_names:
                    metrics["ilisi"] = compute_ilisi(A, batch_key=aux.get("batch_key", "batch"))
                if "kbet" in self.metric_names:
                    metrics["kbet"] = compute_kbet(A, batch_key=aux.get("batch_key", "batch"))
        return EvalReport(metrics=metrics, diagnostics={})

    def compute_label_transfer(self, Z_src: Tensor, labels_src: Tensor, Z_tgt: Tensor, k: int = 10) -> float:
        D = torch.cdist(Z_tgt, Z_src)
        nn = torch.topk(-D, k=min(k, D.shape[1]), dim=1).indices
        votes = labels_src[nn]
        preds = torch.mode(votes, dim=1).values
        vals, counts = torch.unique(preds, return_counts=True)
        return float(counts.float().max() / preds.numel())