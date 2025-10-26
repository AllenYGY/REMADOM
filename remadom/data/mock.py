from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    ad = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_anndata() -> None:
    if ad is None:
        raise RuntimeError("anndata is required to generate mock datasets") from _IMPORT_ERROR


def _make_latent(
    n_cells: int,
    latent_dim: int,
    n_clusters: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    centers = rng.normal(scale=2.5, size=(n_clusters, latent_dim))
    cluster_ids = rng.integers(0, n_clusters, size=n_cells)
    z = centers[cluster_ids] + rng.normal(scale=0.6, size=(n_cells, latent_dim))
    return z, cluster_ids


def _generate_rna(z: np.ndarray, n_genes: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.normal(scale=0.4, size=(z.shape[1], n_genes))
    log_rate = (z @ weights) * 0.3 + rng.normal(scale=0.1, size=(z.shape[0], n_genes))
    rate = np.clip(np.exp(log_rate), 1e-4, 12.0)
    counts = rng.poisson(rate)
    return counts.astype(np.float32)


def _generate_atac(z: np.ndarray, n_peaks: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.normal(scale=0.7, size=(z.shape[1], n_peaks))
    logits = (z @ weights) * 0.4 + rng.normal(scale=0.5, size=(z.shape[0], n_peaks))
    prob = 1.0 / (1.0 + np.exp(-logits))
    return rng.binomial(1, prob).astype(np.float32)


def _generate_adt(z: np.ndarray, n_proteins: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.normal(scale=0.5, size=(z.shape[1], n_proteins))
    mu = (z @ weights) * 0.3 + rng.normal(scale=0.2, size=(z.shape[0], n_proteins))
    mu = np.exp(mu)
    values = rng.normal(loc=mu, scale=0.5)
    return np.clip(values, 0.0, None).astype(np.float32)


def _ensure_modality_presence(masks: Dict[str, np.ndarray]) -> None:
    combo = masks["rna"] | masks["atac"] | masks["adt"]
    if not combo.all():
        # ensure at least RNA is observed when no modality present
        masks["rna"] = masks["rna"] | ~combo


def generate_mock_dataset(
    problem_type: str,
    *,
    n_cells: int = 600,
    n_genes: int = 1000,
    n_peaks: int = 5000,
    n_proteins: int = 30,
    latent_dim: int = 16,
    seed: int | None = None,
) -> Tuple["ad.AnnData", Dict[str, Dict[str, str]]]:
    """
    Generate a synthetic AnnData object covering a given multimodal integration scenario.

    Returns
    -------
    adata : AnnData
        Mock dataset with columns ``has_<mod>`` indicating modality coverage.
    keys : Dict[str, Dict[str, str]]
        Mapping suitable for ExperimentConfig.data.source.keys.
    """

    _require_anndata()
    rng = np.random.default_rng(seed)
    z, cluster_ids = _make_latent(n_cells, latent_dim, n_clusters=min(5, max(2, n_cells // 80)), rng=rng)

    masks = {m: np.zeros(n_cells, dtype=bool) for m in ("rna", "atac", "adt")}
    batch_labels = np.zeros(n_cells, dtype=int)
    dataset_labels = np.zeros(n_cells, dtype=int)
    splits = np.array(["train"] * n_cells)

    problem = problem_type.lower()
    if problem == "paired":
        masks["rna"][:] = True
        masks["adt"][:] = True
        n_batches = 2
        batch_labels = rng.integers(0, n_batches, size=n_cells)
        offsets = rng.normal(scale=0.5, size=(n_batches, latent_dim))
        z = z + offsets[batch_labels]
    elif problem == "unpaired":
        half = n_cells // 2
        masks["rna"][:half] = True
        masks["atac"][half:] = True
        batch_labels[half:] = 1
        dataset_labels = batch_labels.copy()
        z[half:] += rng.normal(scale=0.5, size=(n_cells - half, latent_dim))
    elif problem == "bridge":
        masks["rna"][:] = True
        masks["atac"][:] = True
        bridge_count = max(1, int(0.1 * n_cells))
        bridge_mask = np.zeros(n_cells, dtype=bool)
        bridge_mask[:bridge_count] = True
        rng.shuffle(bridge_mask)
        solo = ~bridge_mask
        rna_only = solo & (rng.random(n_cells) < 0.5)
        atac_only = solo & ~rna_only
        masks["rna"][atac_only] = False
        masks["atac"][rna_only] = False
        batch_labels = rna_only.astype(int)
        dataset_labels = batch_labels.copy()
    elif problem == "mosaic":
        masks["rna"] = rng.random(n_cells) < 0.6
        masks["atac"] = rng.random(n_cells) < 0.5
        masks["adt"] = rng.random(n_cells) < 0.4
    elif problem == "prediction":
        masks["rna"][:] = True
        masks["atac"][:] = True
        split_point = int(n_cells * 0.7)
        splits[split_point:] = "eval"
        masks["atac"][split_point:] = False
        batch_labels[split_point:] = 1
        dataset_labels = batch_labels.copy()
    elif problem == "hierarchical":
        groups = np.array_split(np.arange(n_cells), 3)
        masks["rna"][groups[0]] = True
        masks["atac"][groups[0]] = True
        masks["rna"][groups[1]] = True
        masks["adt"][groups[1]] = True
        masks["atac"][groups[2]] = True
        dataset_labels = np.concatenate(
            [
                np.full(len(groups[0]), 0, dtype=int),
                np.full(len(groups[1]), 1, dtype=int),
                np.full(len(groups[2]), 2, dtype=int),
            ]
        )
        batch_labels = dataset_labels.copy()
        offsets = rng.normal(scale=0.6, size=(3, latent_dim))
        z = z + offsets[dataset_labels]
    else:
        raise ValueError(f"Unknown problem type '{problem_type}'")

    _ensure_modality_presence(masks)

    rna = _generate_rna(z, n_genes, rng)
    atac = _generate_atac(z, n_peaks, rng)
    adt = _generate_adt(z, n_proteins, rng)

    for mod, data in [("rna", rna), ("atac", atac), ("adt", adt)]:
        mask = masks.get(mod)
        if mask is None or data.size == 0:
            continue
        if not mask.any():
            continue
        data[~mask] = 0.0

    obs = pd.DataFrame(
        {
            "batch": batch_labels,
            "dataset": dataset_labels,
            "cluster": [f"cluster_{c}" for c in cluster_ids],
            "split": splits,
            "has_rna": masks["rna"],
            "has_atac": masks["atac"],
            "has_adt": masks["adt"],
        }
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(rna.shape[1])])
    adata = ad.AnnData(rna, obs=obs, var=var)
    adata.layers["counts"] = rna.astype(np.float32)
    if masks["atac"].any():
        adata.obsm["X_atac"] = atac
    if masks["adt"].any():
        adata.obsm["X_adt"] = adt
        adata.uns["adt_names"] = [f"protein_{i}" for i in range(adt.shape[1])]

    keys: Dict[str, Dict[str, str]] = {}
    if masks["rna"].any():
        keys["rna"] = {"X": "X"}
    if masks["atac"].any():
        keys["atac"] = {"obsm": "X_atac"}
    if masks["adt"].any():
        keys["adt"] = {"obsm": "X_adt", "uns_key": "adt_names"}
    adata.uns["remadom_mock_keys"] = keys
    return adata, keys
