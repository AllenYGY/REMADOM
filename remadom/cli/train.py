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
from ..utils.seed import seed_everything
from ..utils.serialization import save_yaml

try:  # Optional plotting dependency
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def _log(message: str) -> None:
    """Emit a lightweight CLI log message."""
    print(f"[remadom][train] {message}")


def cli_train(cfg_path: str, overrides: Optional[List[str]] = None) -> int:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"using device: {device}")
    model = build_model(cfg).to(device)
    heads = build_heads(cfg)
    head_schedules = apply_head_schedules(heads, cfg)
    beta_init, beta_schedule = get_beta_schedule(cfg)
    modality_schedules = get_modality_weight_schedules(cfg)
    optimizer, scheduler = build_optimizer(cfg, model)

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
    _maybe_plot_history(history, run_dir / "loss_curve.png")
    return 0


def main():
    ap = argparse.ArgumentParser("remadom-train")
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("overrides", nargs="*", help="OmegaConf-style overrides (optional)")
    args = ap.parse_args()
    return cli_train(args.cfg, args.overrides)


def _maybe_plot_history(history: Dict[str, List[Dict[str, float]]], path: Path) -> None:
    if plt is None:
        _log("matplotlib not available; skipping loss curve plot")
        return
    train_epochs = history.get("train", [])
    if not train_epochs:
        _log("no training history to plot")
        return
    metrics = sorted(train_epochs[0].keys())
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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
