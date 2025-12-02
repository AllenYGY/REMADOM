from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..config.schema import ExperimentConfig
from ..data.loaders import dataloader, load_anndata
from ..typing import Batch
from ..core.vae import MosaicVAE
from .alignment_metrics import compute_alignment_metrics

try:  # optional plotting dependency
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def run_mock_evaluations(
    cfg: ExperimentConfig,
    model: MosaicVAE,
    run_dir: Path,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    eval_cfg = getattr(cfg, "evaluation", None)
    if eval_cfg is None or not getattr(eval_cfg, "enabled", False):
        return {}
    try:
        adata = load_anndata(cfg.data.source.path)
    except Exception:
        return {}

    truth, obs = _extract_truth(adata)
    if not truth:
        return {}

    loader = dataloader(adata, cfg, batch_size=cfg.optim.batch_size, shuffle=False)
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    collect_samples = bool(getattr(eval_cfg, "save_predictions", False))
    sample_store: Dict[str, Dict[str, object]] = {}
    for task in eval_cfg.tasks:
        if task == "paired_imputation":
            results[task] = _paired_imputation(model, loader, truth, device, sample_store if collect_samples else None)
        elif task == "unpaired_imputation":
            results[task] = _missing_imputation(model, loader, truth, device, sample_store if collect_samples else None)
        elif task == "bridge_imputation":
            results[task] = _bridge_imputation(model, loader, truth, obs, device, sample_store if collect_samples else None)
        elif task == "mosaic_imputation":
            results[task] = _mosaic_imputation(model, loader, truth, device, sample_store if collect_samples else None)
        elif task == "prediction_accuracy":
            results[task] = _prediction_imputation(model, loader, truth, obs, device, sample_store if collect_samples else None)
        elif task == "hierarchical_alignment":
            results[task] = _hierarchical_alignment(model, loader, obs, device)
    if results:
        out_path = run_dir / "evaluation.mock.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    if collect_samples and sample_store:
        _save_sample_outputs(sample_store, run_dir)
    return results


def _extract_truth(adata) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    truth: Dict[str, Tensor] = {}
    obs: Dict[str, Tensor] = {}
    if "truth_rna" in adata.layers:
        truth["rna"] = torch.tensor(np.asarray(adata.layers["truth_rna"]), dtype=torch.float32)
    if "truth_atac" in adata.obsm:
        truth["atac"] = torch.tensor(np.asarray(adata.obsm["truth_atac"]), dtype=torch.float32)
    if "truth_adt" in adata.obsm:
        truth["adt"] = torch.tensor(np.asarray(adata.obsm["truth_adt"]), dtype=torch.float32)
    for mod in ("rna", "atac", "adt"):
        col = f"has_{mod}"
        if col in adata.obs:
            obs[col] = torch.tensor(adata.obs[col].astype(bool).values)
    if "dataset" in adata.obs:
        obs["dataset"] = _encode_obs_column(adata.obs["dataset"].values)
    if "batch" in adata.obs:
        obs["batch"] = _encode_obs_column(adata.obs["batch"].values)
    if "split" in adata.obs:
        obs["split_eval"] = torch.tensor((adata.obs["split"].values == "eval"))
    return truth, obs


def _encode_obs_column(values: np.ndarray) -> Tensor:
    arr = np.asarray(values)
    if arr.dtype.kind not in {"i", "u", "b"}:
        _, inv = np.unique(arr, return_inverse=True)
        arr = inv
    return torch.tensor(arr.astype(np.int64), dtype=torch.long)


def _paired_imputation(
    model: MosaicVAE,
    loader: Iterable[Batch],
    truth: Dict[str, Tensor],
    device: torch.device,
    sample_store: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for mod, truth_tensor in truth.items():
        stats = _drop_and_impute(model, loader, mod, truth_tensor, device, sample_store=sample_store)
        metrics.update({f"{mod}_mae": stats["mae"], f"{mod}_rmse": stats["rmse"]})
    return metrics


def _missing_imputation(model, loader, truth, device, sample_store=None):
    metrics: Dict[str, float] = {}
    for mod, truth_tensor in truth.items():
        stats = _impute_missing_only(model, loader, mod, truth_tensor, device, sample_store=sample_store)
        if stats["count"] > 0:
            metrics.update({f"{mod}_mae": stats["mae"], f"{mod}_rmse": stats["rmse"]})
    return metrics


def _bridge_imputation(model, loader, truth, obs, device, sample_store=None):
    metrics: Dict[str, float] = {}
    for mod, truth_tensor in truth.items():
        stats = _impute_missing_only(model, loader, mod, truth_tensor, device, sample_store=sample_store)
        if stats["count"] > 0:
            metrics[f"{mod}_mae"] = stats["mae"]
            metrics[f"{mod}_rmse"] = stats["rmse"]
    return metrics


def _mosaic_imputation(model, loader, truth, device, sample_store=None):
    metrics = _paired_imputation(model, loader, truth, device, sample_store=sample_store)
    missing = _missing_imputation(model, loader, truth, device, sample_store=sample_store)
    for key, value in missing.items():
        metrics[f"missing_{key}"] = value
    return metrics


def _prediction_imputation(model, loader, truth, obs, device, sample_store=None):
    split_mask = obs.get("split_eval")
    if split_mask is None:
        return {}
    metrics: Dict[str, float] = {}
    for mod, truth_tensor in truth.items():
        stats = _impute_missing_only(model, loader, mod, truth_tensor, device, global_mask=split_mask, sample_store=sample_store)
        if stats["count"] > 0:
            metrics[f"{mod}_mae"] = stats["mae"]
            metrics[f"{mod}_rmse"] = stats["rmse"]
    return metrics


def _hierarchical_alignment(model, loader, obs, device):
    z_all: List[Tensor] = []
    labels: List[Tensor] = []
    for batch in loader:
        batch = _to_device(batch, device)
        if batch.dataset_labels is None:
            continue
        enc = model.encode(batch)
        fused = model.fuse_posteriors(enc)
        z_all.append(fused["mu"].detach().cpu())
        labels.append(batch.dataset_labels.detach().cpu())
    if not z_all or not labels:
        return {}
    z = torch.cat(z_all, dim=0)
    lbl = torch.cat(labels, dim=0)
    return compute_alignment_metrics(z, lbl)


def _drop_and_impute(
    model,
    loader,
    mod: str,
    truth: Tensor,
    device: torch.device,
    *,
    sample_store: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, float]:
    stats = {"abs": 0.0, "sq": 0.0, "count": 0}
    for batch in loader:
        if getattr(batch, f"x_{mod}") is None or getattr(batch, f"has_{mod}") is None:
            continue
        keep_mask = getattr(batch, f"has_{mod}").bool()
        if not keep_mask.any():
            continue
        sub = _subset_batch(batch, keep_mask)
        sub = _to_device(sub, device)
        setattr(sub, f"x_{mod}", torch.zeros_like(getattr(sub, f"x_{mod}")))
        setattr(sub, f"has_{mod}", torch.zeros_like(getattr(sub, f"has_{mod}"), dtype=torch.bool))
        if mod == "rna":
            sub.libsize_rna = torch.zeros_like(sub.libsize_rna) if sub.libsize_rna is not None else None
        preds = model.impute(sub, target_modalities=[mod])[mod]
        pred = _decoder_expectation(mod, preds)
        idx = batch.indices[keep_mask] if batch.indices is not None else None
        target = truth[idx] if idx is not None else truth[: pred.shape[0]]
        _accumulate_errors(stats, pred, target.to(pred.device))
        _record_samples(sample_store, mod, pred, target)
    return _finalize_stats(stats)


def _impute_missing_only(
    model,
    loader,
    mod: str,
    truth: Tensor,
    device: torch.device,
    global_mask: Optional[Tensor] = None,
    sample_store: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, float]:
    stats = {"abs": 0.0, "sq": 0.0, "count": 0}
    for batch in loader:
        has = getattr(batch, f"has_{mod}", None)
        if has is None:
            continue
        missing = ~has.bool()
        if global_mask is not None and batch.indices is not None:
            gm = global_mask[batch.indices.cpu()]
            missing = missing & gm
        if not missing.any():
            continue
        sub = _subset_batch(batch, missing)
        sub = _to_device(sub, device)
        preds = model.impute(sub, target_modalities=[mod])[mod]
        pred = _decoder_expectation(mod, preds)
        idx = batch.indices[missing] if batch.indices is not None else None
        target = truth[idx] if idx is not None else truth[: pred.shape[0]]
        _accumulate_errors(stats, pred, target.to(pred.device))
        _record_samples(sample_store, mod, pred, target)
    return _finalize_stats(stats)


def _subset_batch(batch: Batch, mask: Tensor) -> Batch:
    data = {}
    for field in fields(Batch):
        val = getattr(batch, field.name)
        if isinstance(val, torch.Tensor) and val is not None:
            data[field.name] = val[mask]
        else:
            data[field.name] = val
    return Batch(**data)  # type: ignore[arg-type]


def _to_device(batch: Batch, device: torch.device) -> Batch:
    for field in fields(Batch):
        val = getattr(batch, field.name)
        if isinstance(val, torch.Tensor) and val is not None:
            setattr(batch, field.name, val.to(device))
    return batch


def _decoder_expectation(mod: str, params: Dict[str, Tensor]) -> Tensor:
    if mod == "rna":
        return params["mu"]
    if mod == "atac":
        logits = params.get("logits")
        if logits is None:
            raise KeyError("ATAC decoder outputs logits")
        return torch.sigmoid(logits)
    if mod == "adt":
        return params["mu"]
    raise KeyError(f"Unsupported modality '{mod}' for evaluation")


def _accumulate_errors(stats: Dict[str, float], pred: Tensor, truth: Tensor) -> None:
    diff = (pred - truth).detach()
    stats["abs"] += float(diff.abs().sum().item())
    stats["sq"] += float(diff.pow(2).sum().item())
    stats["count"] += int(diff.numel())


def _finalize_stats(stats: Dict[str, float]) -> Dict[str, float]:
    if stats["count"] == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "count": 0}
    mae = stats["abs"] / stats["count"]
    rmse = (stats["sq"] / stats["count"]) ** 0.5
    return {"mae": mae, "rmse": rmse, "count": stats["count"]}


def _record_samples(
    store: Optional[Dict[str, Dict[str, object]]],
    mod: str,
    pred: Tensor,
    truth: Tensor,
    *,
    max_points: int = 50000,
) -> None:
    if store is None:
        return
    flat_pred = pred.detach().cpu().reshape(-1)
    flat_truth = truth.detach().cpu().reshape(-1)
    if flat_pred.numel() == 0 or flat_truth.numel() == 0:
        return
    entry = store.setdefault(mod, {"pred": [], "truth": [], "count": 0})
    count = entry["count"]  # type: ignore[assignment]
    remaining = max(0, max_points - int(count))
    if remaining == 0:
        return
    if flat_pred.numel() > remaining:
        idx = torch.randperm(flat_pred.numel())[:remaining]
        flat_pred = flat_pred[idx]
        flat_truth = flat_truth[idx]
    entry["pred"].append(flat_pred)  # type: ignore[attr-defined]
    entry["truth"].append(flat_truth)  # type: ignore[attr-defined]
    entry["count"] = int(count) + flat_pred.numel()  # type: ignore[assignment]


def _save_sample_outputs(sample_store: Dict[str, Dict[str, object]], run_dir: Path) -> None:
    arrays: Dict[str, np.ndarray] = {}
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for mod, entry in sample_store.items():
        preds = entry.get("pred") or []
        truths = entry.get("truth") or []
        if not preds or not truths:
            continue
        pred_cat = torch.cat(preds, dim=0).numpy()  # type: ignore[arg-type]
        truth_cat = torch.cat(truths, dim=0).numpy()  # type: ignore[arg-type]
        arrays[f"{mod}_pred"] = pred_cat
        arrays[f"{mod}_truth"] = truth_cat
        plot_data[mod] = (truth_cat, pred_cat)
    if arrays:
        np.savez(run_dir / "evaluation_samples.npz", **arrays)
        _plot_sample_distributions(plot_data, run_dir / "evaluation_plots.png")


def _plot_sample_distributions(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    if not data or plt is None:
        return
    mods = sorted(data.keys())
    fig, axes = plt.subplots(len(mods), 2, figsize=(10, 3 * len(mods)), dpi=150, squeeze=False)
    for idx, mod in enumerate(mods):
        truth, pred = data[mod]
        sample = min(4000, truth.shape[0])
        if sample < truth.shape[0]:
            idxs = np.random.choice(truth.shape[0], sample, replace=False)
            truth_sample = truth[idxs]
            pred_sample = pred[idxs]
        else:
            truth_sample = truth
            pred_sample = pred
        ax_scatter = axes[idx, 0]
        ax_hist = axes[idx, 1]
        min_val = min(truth_sample.min(), pred_sample.min())
        max_val = max(truth_sample.max(), pred_sample.max())
        ax_scatter.scatter(truth_sample, pred_sample, s=4, alpha=0.3)
        ax_scatter.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=0.8)
        ax_scatter.set_title(f"{mod.upper()} predicted vs true")
        ax_scatter.set_xlabel("True")
        ax_scatter.set_ylabel("Predicted")
        residual = pred_sample - truth_sample
        ax_hist.hist(residual, bins=50, alpha=0.8, color="tab:blue")
        ax_hist.set_title(f"{mod.upper()} residual distribution")
        ax_hist.set_xlabel("Prediction - True")
        ax_hist.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
