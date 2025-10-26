from __future__ import annotations

import numpy as np

from remadom.data.mock import generate_mock_dataset


def _check_common(adata, keys):
    assert "batch" in adata.obs
    for mod, cfg in keys.items():
        assert f"has_{mod}" in adata.obs
        mask = adata.obs[f"has_{mod}"].values
        assert mask.dtype == bool
        assert mask.any(), f"modality {mod} should be present in at least one cell"
        if mod == "rna":
            assert adata.X.shape[0] == mask.shape[0]
        elif mod == "atac":
            assert "X_atac" in adata.obsm
            assert adata.obsm["X_atac"].shape[0] == mask.shape[0]
        elif mod == "adt":
            assert "X_adt" in adata.obsm
            assert adata.obsm["X_adt"].shape[0] == mask.shape[0]


def test_generate_mock_mosaic():
    adata, keys = generate_mock_dataset("mosaic", n_cells=120, n_genes=50, n_peaks=80, n_proteins=12, seed=42)
    _check_common(adata, keys)
    assert 0.2 < adata.obs["has_rna"].mean() < 1.0
    assert 0.1 < adata.obs["has_atac"].mean() < 1.0


def test_generate_mock_bridge():
    adata, keys = generate_mock_dataset("bridge", n_cells=120, n_genes=40, n_peaks=60, seed=7)
    _check_common(adata, keys)
    assert set(keys.keys()) == {"rna", "atac"}
    both = adata.obs["has_rna"] & adata.obs["has_atac"]
    assert both.any()


def test_prediction_split_labels():
    adata, keys = generate_mock_dataset("prediction", n_cells=90, n_genes=30, n_peaks=50, seed=3)
    _check_common(adata, keys)
    assert "split" in adata.obs
    assert set(adata.obs["split"]) == {"train", "eval"}
