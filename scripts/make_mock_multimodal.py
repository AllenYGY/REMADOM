#!/usr/bin/env python3
"""Generate mock multimodal AnnData datasets for REMADOM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from remadom.data.mock import generate_mock_dataset


def _keys_to_serialisable(keys: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {mod: dict(cfg) for mod, cfg in keys.items()}


def _build_default_config(path: Path, keys: Dict[str, Dict[str, str]], adata_dims: Dict[str, int]) -> Dict[str, Any]:
    encoders: Dict[str, Any] = {}
    decoders: Dict[str, Any] = {}
    for mod, dims in adata_dims.items():
        if dims is None:
            continue
        if mod == "rna":
            encoders[mod] = {"in_dim": dims, "hidden_dims": [256, 256]}
            decoders[mod] = {"out_dim": dims, "dispersion": "gene", "library": True}
        else:
            encoders[mod] = {"in_dim": dims, "hidden_dims": [256, 256]}
            decoders[mod] = {"out_dim": dims, "library": False}
            if mod == "adt":
                decoders[mod]["hidden_dims"] = [128, 128]
    config: Dict[str, Any] = {
        "data": {
            "source": {
                "path": str(path),
                "batch_key": "batch",
                "keys": _keys_to_serialisable(keys),
            }
        },
        "model": {
            "latent_bio": 16,
            "encoders": encoders,
            "decoders": decoders,
            "beta": 1.0,
        },
        "optim": {
            "epochs": 10,
            "batch_size": 256,
            "lr": 0.001,
            "amp": {"enabled": True, "dtype": "bf16"},
        },
        "logging": {"run_dir": "runs/mock"},
    }
    return config


def main() -> int:
    parser = argparse.ArgumentParser("make_mock_multimodal")
    parser.add_argument("--problem", required=True, help="Problem type (paired, unpaired, bridge, mosaic, prediction, hierarchical)")
    parser.add_argument("--out", required=True, help="Path to output .h5ad file")
    parser.add_argument("--config-out", help="Optional path to write a ready-to-edit YAML config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cells", type=int, default=600)
    parser.add_argument("--genes", type=int, default=1000)
    parser.add_argument("--peaks", type=int, default=5000)
    parser.add_argument("--proteins", type=int, default=30)
    args = parser.parse_args()

    adata, keys = generate_mock_dataset(
        args.problem,
        n_cells=args.cells,
        n_genes=args.genes,
        n_peaks=args.peaks,
        n_proteins=args.proteins,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"Saved dataset to {out_path}")

    if args.config_out:
        if yaml is None:
            raise RuntimeError("PyYAML is required to emit config files; install pyyaml or omit --config-out")
        dims = {
            "rna": int(adata.n_vars) if "rna" in keys else None,
            "atac": int(adata.obsm["X_atac"].shape[1]) if "atac" in keys else None,
            "adt": int(adata.obsm["X_adt"].shape[1]) if "adt" in keys else None,
        }
        config = _build_default_config(out_path, keys, dims)
        config_path = Path(args.config_out)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        print(f"Wrote config template to {config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
