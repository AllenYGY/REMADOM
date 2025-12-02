from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config.factories import (
    apply_head_schedules,
    build_heads,
    build_model,
    build_optimizer,
    get_beta_schedule,
    get_modality_weight_schedules,
)
from ..config.resolve import resolve_config
from ..config.schema import ExperimentConfig
from ..data.loaders import build_dataloaders
from ..train.trainer import Trainer
from ..eval.mock_eval import run_mock_evaluations
from ..eval.plots import latent_umap, latent_tsne
from ..eval.alignment_metrics import compute_batch_metrics
from ..utils.seed import seed_everything
from ..utils.serialization import save_yaml
from ..utils.checkpoint import resume_trainer
from ..data.loaders import dataloader_from_source

try:  # Optional plotting dependency
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def _log(message: str) -> None:
    """Emit a lightweight CLI log message."""
    print(f"[remadom][train] {message}")


def cli_train(
    cfg_path: str,
    overrides: Optional[List[str]] = None,
    *,
    force_cpu: bool = False,
    plot: bool = True,
    metrics_only: bool = False,
    plot_latent: bool = False,
    plot_umap: bool = False,
    plot_tsne: bool = False,
    resume: Optional[str] = None,
) -> int:
    _log(f"loading config: {cfg_path}")
    cfg = resolve_config([cfg_path] + (overrides or []))
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError("resolve_config must return an ExperimentConfig")

    if cfg.seed is not None:
        _log(f"seeding RNGs with seed={cfg.seed}")
        seed_everything(cfg.seed)

    cfg_stem = Path(cfg_path).stem
    base_dir = Path(cfg.logging.run_dir)
    run_dir = base_dir if cfg_stem in base_dir.parts else base_dir / cfg_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"artifacts will be stored in: {run_dir.resolve()}")
    save_yaml(cfg.model_dump(), run_dir / "config.resolved.yaml")

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"using device: {device}")
    model = build_model(cfg).to(device)
    heads = build_heads(cfg)
    head_schedules = apply_head_schedules(heads, cfg)
    beta_init, beta_schedule = get_beta_schedule(cfg)
    modality_schedules = get_modality_weight_schedules(cfg)
    optimizer, scheduler = build_optimizer(cfg, model)

    if resume:
        _log(f"resuming from checkpoint: {resume}")
        trainer = Trainer(
            model,
            optimizer,
            scheduler=scheduler,
            heads=heads,
            cfg=cfg,
        )
        resume_trainer(trainer, resume)
        # ensure schedules available after resume
        trainer.head_schedules = head_schedules
        trainer._beta_schedule = beta_schedule
        trainer._beta_value = float(beta_init)
        trainer._modality_schedules = modality_schedules
    else:
        trainer = Trainer(
            model,
            optimizer,
            scheduler=scheduler,
            heads=heads,
            cfg=cfg,
            head_schedules=head_schedules,
            beta_schedule=beta_schedule,
            beta_init=beta_init,
            modality_schedules=modality_schedules,
        )

    train_loader, val_loader, _ = build_dataloaders(cfg)
    epochs = cfg.optim.epochs
    batch_size = cfg.optim.batch_size
    _log(f"starting training for {epochs} epochs with batch_size={batch_size}")

    trainer = Trainer(
        model,
        optimizer,
        scheduler=scheduler,
        heads=heads,
        cfg=cfg,
        head_schedules=head_schedules,
        beta_schedule=beta_schedule,
        beta_init=beta_init,
        modality_schedules=modality_schedules,
    )

    history = trainer.fit(train_loader, val_loader)
    _log("training finished")

    if not metrics_only:
        trainer.save_checkpoint(run_dir / "checkpoint.last.pt")
        _log("saved latest checkpoint: checkpoint.last.pt")
        if trainer.best_state is not None:
            best_path = run_dir / "checkpoint.best.pt"
            torch.save({"model": trainer.best_state}, best_path)
            _log("saved best-validation checkpoint: checkpoint.best.pt")

        history_path = run_dir / "train_history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        _log(f"training history written to: {history_path}")
        if plot:
            _maybe_plot_history(history, run_dir / "loss_curve.png")
    elif plot:
        _maybe_plot_history(history, run_dir / "loss_curve.png")
    summary_path = run_dir / "metrics.final.json"
    summary = _summarize_metrics(history, trainer)
    eval_results = run_mock_evaluations(cfg, trainer.model, run_dir, device)
    if eval_results:
        summary["evaluation"] = eval_results
        _log(f"mock evaluation metrics written to: {run_dir / 'evaluation.mock.json'}")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _log(f"final metrics written to: {summary_path}")
    _log_final_snapshot(summary)
    _maybe_write_bridge_details(summary, run_dir)
    _maybe_write_alignment_metrics(summary, run_dir)
    _maybe_plot_bridge(summary, run_dir, plot=plot)
    if plot_latent:
        _maybe_plot_latent(trainer, run_dir)
    if plot_umap:
        _maybe_plot_embedded(trainer, run_dir, kind="umap")
    if plot_tsne:
        _maybe_plot_embedded(trainer, run_dir, kind="tsne")
    # Optional SCIB metrics if dependencies present
    scib_metrics = _maybe_compute_scib(cfg, trainer, device)
    if scib_metrics:
        summary["scib"] = scib_metrics
        scib_path = run_dir / "scib_metrics.json"
        with scib_path.open("w", encoding="utf-8") as f:
            json.dump(scib_metrics, f, indent=2)
        _log(f"SCIB metrics written to: {scib_path}")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return 0


def main():
    ap = argparse.ArgumentParser("remadom-train")
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--force-cpu", action="store_true", help="Run on CPU even if CUDA is available")
    ap.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    ap.add_argument("--metrics-only", action="store_true", help="Skip checkpoint/history and only emit metrics")
    ap.add_argument("--plot-latent", action="store_true", help="Plot latent PCA scatter (if matplotlib available)")
    ap.add_argument("--plot-umap", action="store_true", help="Plot latent UMAP (requires umap-learn)")
    ap.add_argument("--plot-tsne", action="store_true", help="Plot latent t-SNE (requires scikit-learn)")
    ap.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    ap.add_argument("overrides", nargs="*", help="OmegaConf-style overrides (optional)")
    args = ap.parse_args()
    return cli_train(
        args.cfg,
        args.overrides,
        force_cpu=args.force_cpu,
        plot=not args.no_plot,
        metrics_only=args.metrics_only,
        plot_latent=args.plot_latent,
        plot_umap=args.plot_umap,
        plot_tsne=args.plot_tsne,
        resume=args.resume,
    )


def _maybe_plot_history(history: Dict[str, List[Dict[str, float]]], path: Path) -> None:
    if plt is None:
        _log("matplotlib not available; skipping loss curve plot")
        return
    train_epochs = history.get("train", [])
    if not train_epochs:
        _log("no training history to plot")
        return
    metrics = _select_plot_metrics(train_epochs[0])
    if not metrics:
        _log("no loss-like metrics found for plotting")
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    epochs = range(1, len(train_epochs) + 1)
    for metric in metrics:
        values = [ep.get(metric, float("nan")) for ep in train_epochs]
        ax.plot(epochs, values, label=f"train/{metric}")
    val_epochs = history.get("val") or []
    if val_epochs:
        val_epochs = val_epochs[: len(epochs)]
        for metric in metrics:
            values = [ep.get(metric, float("nan")) for ep in val_epochs]
            ax.plot(
                range(1, len(values) + 1),
                values,
                label=f"val/{metric}",
                linestyle="--",
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    _log(f"loss curves plotted to: {path}")


def _select_plot_metrics(epoch_metrics: Dict[str, float]) -> List[str]:
    preferred_tokens = ("loss", "recon", "kl", "head")
    keys: List[str] = [k for k in epoch_metrics if any(token in k for token in preferred_tokens)]
    if not keys:
        keys = list(epoch_metrics.keys())
    return sorted(keys)


def _summarize_metrics(history: Dict[str, List[Dict[str, float]]], trainer: Trainer) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    if history.get("train"):
        summary["train"] = history["train"][-1]
    if history.get("val"):
        summary["val"] = history["val"][-1]
    head_details = getattr(trainer, "_latest_head_details", {})
    if head_details:
        summary["head_details"] = head_details
    head_trace = getattr(trainer, "_latest_head_loss_trace", {})
    if head_trace:
        summary["head_loss_trace"] = head_trace
    align_metrics = getattr(trainer, "_latest_alignment_metrics", {})
    if align_metrics:
        summary["alignment_metrics"] = align_metrics
    return summary


def _log_final_snapshot(summary: Dict[str, object]) -> None:
    train_metrics = summary.get("train")
    if isinstance(train_metrics, dict):
        display_keys = [k for k in train_metrics.keys() if k in ("loss", "recon", "kl", "head_total") or k.endswith("_loss")]
        display_keys = display_keys[:5] if len(display_keys) > 5 else display_keys
        formatted = ", ".join(f"{k}={train_metrics[k]:.4f}" for k in display_keys if isinstance(train_metrics.get(k), (int, float)))
        if formatted:
            _log(f"final train metrics: {formatted}")
    val_metrics = summary.get("val")
    if isinstance(val_metrics, dict):
        display_keys = [k for k in val_metrics.keys() if k in ("loss", "recon", "kl", "head_total")]
        formatted = ", ".join(f"{k}={val_metrics[k]:.4f}" for k in display_keys if isinstance(val_metrics.get(k), (int, float)))
        if formatted:
            _log(f"final val metrics: {formatted}")
    align = summary.get("alignment_metrics")
    if isinstance(align, dict) and align:
        train_align = align.get("train")
        if isinstance(train_align, dict) and train_align:
            snippet = ", ".join(f"{k}={v:.4f}" for k, v in list(train_align.items())[:3])
            _log(f"alignment diagnostics: {snippet}")


def _maybe_write_bridge_details(summary: Dict[str, object], run_dir: Path) -> None:
    details = summary.get("head_details")
    if not isinstance(details, dict):
        return
    train_details = details.get("train")
    if not isinstance(train_details, dict):
        return
    bridge_entries = {k: v for k, v in train_details.items() if k.startswith("bridge")}
    if not bridge_entries:
        return
    path = run_dir / "bridge_metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(bridge_entries, f, indent=2)
    _log(f"bridge diagnostics written to: {path}")


def _maybe_write_alignment_metrics(summary: Dict[str, object], run_dir: Path) -> None:
    align = summary.get("alignment_metrics")
    if not isinstance(align, dict) or not align:
        return
    path = run_dir / "alignment_metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(align, f, indent=2)
    _log(f"alignment metrics written to: {path}")


def _maybe_plot_bridge(summary: Dict[str, object], run_dir: Path, *, plot: bool) -> None:
    if not plot or plt is None:
        return
    details = summary.get("head_details")
    if not isinstance(details, dict):
        return
    train_details = details.get("train")
    if not isinstance(train_details, dict):
        return
    bridge_entries = train_details.get("bridge_mnn") or train_details.get("bridge_seeded") or train_details.get("bridge_dictionary")
    if not isinstance(bridge_entries, dict):
        return
    pairs = bridge_entries.get("bridge_pairs")
    if not isinstance(pairs, list) or not pairs:
        return
    labels = [str(item.get("pair", "?")) for item in pairs]
    values = [item.get("edges", 0) for item in pairs]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.bar(labels, values)
    ax.set_ylabel("Edges")
    ax.set_title("Bridge edges per cohort pair")
    fig.tight_layout()
    out = run_dir / "bridge_edges.png"
    fig.savefig(out)
    plt.close(fig)
    _log(f"bridge edge plot written to: {out}")


def _maybe_plot_latent(trainer: Trainer, run_dir: Path) -> None:
    if plt is None:
        _log("matplotlib not available; skipping latent plot")
        return
    z = getattr(trainer, "_latest_z", None)
    groups = getattr(trainer, "_latest_groups", None)
    if z is None or not isinstance(z, torch.Tensor):
        _log("no latent cached for plotting")
        return
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        _log("numpy not available; skipping latent plot")
        return
    z_np = z.detach().cpu().numpy()
    color = None
    if isinstance(groups, torch.Tensor):
        color = groups.detach().cpu().numpy()
    z_center = z_np - z_np.mean(0, keepdims=True)
    cov = z_center.T @ z_center
    vals, vecs = np.linalg.eigh(cov)
    comp = vecs[:, -2:]
    proj = z_np @ comp
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=color, s=8, cmap="tab10" if color is not None else None, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent PCA")
    if color is not None:
        plt.colorbar(sc, ax=ax, label="group")
    fig.tight_layout()
    out = run_dir / "latent_pca.png"
    fig.savefig(out)
    plt.close(fig)
    _log(f"latent plot written to: {out}")


def _maybe_plot_embedded(trainer: Trainer, run_dir: Path, *, kind: str = "umap") -> None:
    z = getattr(trainer, "_latest_z", None)
    groups = getattr(trainer, "_latest_groups", None)
    if z is None or not isinstance(z, torch.Tensor):
        _log(f"no latent cached for plotting {kind}")
        return
    z_np = z.detach().cpu().numpy()
    color = groups.detach().cpu().numpy() if isinstance(groups, torch.Tensor) else None
    out = run_dir / f"latent_{kind}.png"
    if kind == "umap":
        path = latent_umap(z_np, groups=color, path=str(out))
    else:
        path = latent_tsne(z_np, groups=color, path=str(out))
    if path:
        _log(f"{kind} plot written to: {path}")
    else:
        _log(f"skipping {kind} plot (dependency missing)")


def _maybe_compute_scib(cfg: ExperimentConfig, trainer: Trainer, device: torch.device) -> Dict[str, float]:
    try:
        import anndata  # noqa: F401
    except Exception:
        return {}
    if not getattr(cfg.logging, "collect_metrics", False):
        return {}
    # Rebuild a small loader without shuffle to get embeddings
    loader, _ = dataloader_from_source(
        cfg.data.source,
        registry=None,
        batch_size=min(cfg.optim.batch_size, 2048),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = trainer.model
    model.eval()
    zs = []
    Xs = []
    groups = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = trainer._to_device(batch)  # type: ignore[attr-defined]
            enc = model.encode(batch)
            fused = model.fuse_posteriors(enc)
            zs.append(fused["mu"].cpu())
            if batch.x_rna is not None:
                Xs.append(batch.x_rna.cpu())
            group = batch.batch_labels if batch.batch_labels is not None else batch.dataset_labels
            if group is not None:
                groups.append(group.cpu())
            if len(zs) * cfg.optim.batch_size > 4000:  # cap for speed
                break
    if not zs or not Xs or not groups:
        return {}
    z_cat = torch.cat(zs, dim=0)
    X_cat = torch.cat(Xs, dim=0)
    g_cat = torch.cat(groups, dim=0)
    return compute_batch_metrics(z_cat, X_cat, g_cat)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
