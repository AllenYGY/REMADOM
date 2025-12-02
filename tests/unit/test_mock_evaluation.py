from __future__ import annotations

from pathlib import Path

import torch

from remadom.config.schema import (
    DataConfig,
    DataSource,
    DecoderConfig,
    EncoderConfig,
    EvaluationConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
)
from remadom.config.factories import build_model
from remadom.data.mock import generate_mock_dataset
from remadom.eval.mock_eval import run_mock_evaluations


def _make_cfg(tmp_path: Path):
    adata, keys = generate_mock_dataset("paired", n_cells=48, n_genes=64, n_peaks=32, n_proteins=16, seed=1)
    data_path = tmp_path / "mock_paired.h5ad"
    adata.write_h5ad(data_path)
    encoders = {}
    if "rna" in keys:
        encoders["rna"] = EncoderConfig(in_dim=adata.shape[1], hidden_dims=(64, 32))
    if "atac" in keys:
        encoders["atac"] = EncoderConfig(in_dim=adata.obsm["X_atac"].shape[1], hidden_dims=(64, 32))
    if "adt" in keys:
        encoders["adt"] = EncoderConfig(in_dim=adata.obsm["X_adt"].shape[1], hidden_dims=(64, 32))
    decoders = {mod: DecoderConfig(out_dim=encoders[mod].in_dim, library=(mod == "rna")) for mod in encoders}
    cfg = ExperimentConfig(
        data=DataConfig(source=DataSource(path=str(data_path), keys=keys, batch_key="batch")),
        model=ModelConfig(latent_bio=8, encoders=encoders, decoders=decoders, beta=1.0),
        logging=LoggingConfig(run_dir=str(tmp_path / "runs")),
        evaluation=EvaluationConfig(enabled=True, tasks=["paired_imputation"]),
    )
    return cfg


def test_run_mock_evaluations(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    model = build_model(cfg)
    device = torch.device("cpu")
    summary = run_mock_evaluations(cfg, model, tmp_path, device)
    assert "paired_imputation" in summary
    out_file = tmp_path / "evaluation.mock.json"
    assert out_file.exists()
