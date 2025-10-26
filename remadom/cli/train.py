from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

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


def cli_train(cfg_path: str, overrides: Optional[List[str]] = None) -> int:
    cfg = resolve_config([cfg_path] + (overrides or []))
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError("resolve_config must return an ExperimentConfig")

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    run_dir = Path(cfg.logging.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg.model_dump(), run_dir / "config.resolved.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    heads = build_heads(cfg)
    head_schedules = apply_head_schedules(heads, cfg)
    beta_init, beta_schedule = get_beta_schedule(cfg)
    modality_schedules = get_modality_weight_schedules(cfg)
    optimizer, scheduler = build_optimizer(cfg, model)

    train_loader, val_loader, _ = build_dataloaders(cfg)

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

    trainer.save_checkpoint(run_dir / "checkpoint.last.pt")
    if trainer.best_state is not None:
        best_path = run_dir / "checkpoint.best.pt"
        torch.save({"model": trainer.best_state}, best_path)

    history_path = run_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return 0


def main():
    ap = argparse.ArgumentParser("remadom-train")
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("overrides", nargs="*", help="OmegaConf-style overrides (optional)")
    args = ap.parse_args()
    return cli_train(args.cfg, args.overrides)
