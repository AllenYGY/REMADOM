from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch

from ..adapters.mappers import FixedMixtureMapper
from ..config.factories import build_model
from ..config.resolve import resolve_config
from ..data.loaders import (
    build_registry_from_adata,
    dataloader_from_anndata,
    load_anndata,
)
from ..train.trainer import Trainer


def _load_model(cfg, ckpt: str, device: torch.device):
    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model_state = state.get("model") or state.get("model_state") or state
    model.load_state_dict(model_state)
    model.eval()
    return model


def _embed_all(model, cfg, adata, registry, device: torch.device) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    loader = dataloader_from_anndata(
        adata,
        keys=cfg.data.source.keys,
        batch_key=cfg.data.source.batch_key,
        registry=registry,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(model, opt, cfg=cfg)
    zs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = trainer._to_device(batch)
            z = model.embed(batch)["z"].detach().cpu().numpy()
            zs.append(z)
            if batch.batch_labels is not None:
                labels.append(batch.batch_labels.detach().cpu().numpy())
    Z = np.vstack(zs) if zs else np.zeros((0, getattr(model, "latent_dim", 0)))
    y = np.hstack(labels) if len(labels) > 0 else None
    return Z, y


def cli_mapref(
    ref_cfg_path: str,
    ref_ckpt: str,
    query_cfg_path: Optional[str],
    output_dir: str,
    query_data_path: Optional[str] = None,
) -> int:
    """
    Map a query dataset into the reference latent space.

    Steps:
      1) Load reference cfg + checkpoint, embed reference, fit FixedMixtureMapper on batch labels.
      2) Load query cfg (or reuse ref cfg) and embed using the same model/registry.
      3) Map query embeddings to reference labels and save artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_cfg = resolve_config([ref_cfg_path])
    query_cfg = resolve_config([query_cfg_path]) if query_cfg_path else ref_cfg

    # Load datasets
    adata_ref = load_anndata(ref_cfg.data.source.path)
    if query_data_path:
        adata_query = load_anndata(query_data_path)
    else:
        adata_query = load_anndata(query_cfg.data.source.path)

    # Build shared registry from reference
    registry = build_registry_from_adata(adata_ref, ref_cfg.data.source.keys)

    # Load model and embed
    model = _load_model(ref_cfg, ref_ckpt, device)
    Z_ref, y_ref = _embed_all(model, ref_cfg, adata_ref, registry, device)
    mapper = FixedMixtureMapper()
    if y_ref is None:
        y_ref = np.zeros(Z_ref.shape[0], dtype=int)
    mapper.fit(torch.from_numpy(Z_ref).float(), torch.from_numpy(y_ref).long())

    Z_q, _ = _embed_all(model, query_cfg, adata_query, registry, device)
    preds, probs = mapper.map(torch.from_numpy(Z_q).float(), return_probs=True)

    # Persist artifacts
    np.save(os.path.join(output_dir, "Z_ref.npy"), Z_ref)
    np.save(os.path.join(output_dir, "labels_ref.npy"), y_ref)
    np.save(os.path.join(output_dir, "Z_query.npy"), Z_q)
    np.save(os.path.join(output_dir, "preds_query.npy"), preds.detach().cpu().numpy())
    if probs is not None:
        np.save(os.path.join(output_dir, "probs_query.npy"), probs.detach().cpu().numpy())
    print(f"[mapref] artifacts saved to: {output_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser("remadom-mapref")
    ap.add_argument("--ref-cfg", required=True, help="Reference config yaml")
    ap.add_argument("--ref-ckpt", required=True, help="Reference checkpoint (from training)")
    ap.add_argument("--query-cfg", help="Query config yaml; defaults to ref cfg")
    ap.add_argument("--query-data", help="Override query data path (h5ad)")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    return cli_mapref(args.ref_cfg, args.ref_ckpt, args.query_cfg, args.out, query_data_path=args.query_data)
