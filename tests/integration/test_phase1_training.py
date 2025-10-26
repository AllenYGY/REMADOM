from __future__ import annotations

from pathlib import Path

import pytest

from remadom.config.schema import (
    AmpConfig,
    DataConfig,
    DataSource,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
)
from remadom.config.factories import (
    apply_head_schedules,
    build_heads,
    build_model,
    build_optimizer,
    get_beta_schedule,
    get_modality_weight_schedules,
)
from remadom.data.loaders import build_dataloaders
from remadom.data.mock import generate_mock_dataset
from remadom.train.trainer import Trainer


@pytest.mark.parametrize("problem", ["mosaic", "bridge"])
def test_phase1_training_smoke(tmp_path: Path, problem: str) -> None:
    adata, keys = generate_mock_dataset(problem, n_cells=120, n_genes=48, n_peaks=64, n_proteins=10, seed=123)
    path = tmp_path / f"mock_{problem}.h5ad"
    adata.write_h5ad(path)

    encoders = {}
    decoders = {}
    if "rna" in keys:
        encoders["rna"] = EncoderConfig(in_dim=adata.n_vars, hidden_dims=(128, 128), dropout=0.0)
        decoders["rna"] = DecoderConfig(out_dim=adata.n_vars, hidden_dims=(128, 128), dispersion="gene", library=True)
    if "atac" in keys:
        encoders["atac"] = EncoderConfig(in_dim=adata.obsm["X_atac"].shape[1], hidden_dims=(128, 128))
        decoders["atac"] = DecoderConfig(out_dim=adata.obsm["X_atac"].shape[1], hidden_dims=(128, 128), library=False)
    if "adt" in keys:
        encoders["adt"] = EncoderConfig(in_dim=adata.obsm["X_adt"].shape[1], hidden_dims=(64, 64))
        decoders["adt"] = DecoderConfig(out_dim=adata.obsm["X_adt"].shape[1], hidden_dims=(64, 64), library=False)

    cfg = ExperimentConfig(
        data=DataConfig(
            source=DataSource(path=str(path), keys=keys, batch_key="batch"),
            valid=None,
        ),
        model=ModelConfig(latent_bio=8, encoders=encoders, decoders=decoders, beta=1.0),
        optim=OptimConfig(
            epochs=2,
            batch_size=64,
            lr=5e-4,
            weight_decay=0.0,
            amp=AmpConfig(enabled=False, dtype="bf16"),
        ),
    )

    model = build_model(cfg)
    heads = build_heads(cfg)
    head_schedules = apply_head_schedules(heads, cfg)
    beta_init, beta_sched = get_beta_schedule(cfg)
    modality_schedules = get_modality_weight_schedules(cfg)
    optimizer, scheduler = build_optimizer(cfg, model)

    train_loader, val_loader, _ = build_dataloaders(cfg, batch_size=cfg.optim.batch_size)
    trainer = Trainer(
        model,
        optimizer,
        scheduler=scheduler,
        heads=heads,
        cfg=cfg,
        head_schedules=head_schedules,
        beta_schedule=beta_sched,
        beta_init=beta_init,
        modality_schedules=modality_schedules,
    )
    history = trainer.fit(train_loader, val_loader)

    assert len(history["train"]) == cfg.optim.epochs
    assert trainer.best_state is None  # no val loader provided
    last_loss = history["train"][-1]["loss"]
    assert last_loss == pytest.approx(last_loss)  # finite
