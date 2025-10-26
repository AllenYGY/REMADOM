from __future__ import annotations
import argparse
import os
import numpy as np
import torch
from ..config.resolve import resolve_config
from ..config.factories import build_model
from ..data.loaders import load_anndata, dataloader
from ..adapters.mappers import FixedMixtureMapper
from ..adapters.arches import Adapter

def cli_mapref(cfg_path: str, checkpoint_ref: str, checkpoint_query: str | None, output_dir: str) -> int:
    """
    Map query dataset into a reference latent space:
    - Load reference config + checkpoint; embed reference; fit mixture mapper.
    - Load query; embed with optional small residual adapter finetuning; map to ref labels.
    """
    cfg = resolve_config([cfg_path])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load reference model
    model_ref = build_model(cfg).to(device)
    state_ref = torch.load(checkpoint_ref, map_location=device)
    model_ref.load_state_dict(state_ref["model_state"])
    model_ref.eval()

    # Embed reference
    adata_ref = load_anndata(cfg.data.path)  # assuming same path; in practice use cfg.data.ref_path
    loader_ref = dataloader(adata_ref, cfg, batch_size=cfg.optim.batch_size, shuffle=False)
    zs = []
    batches = []
    with torch.no_grad():
        for batch in loader_ref:
            from ..train.trainer import Trainer
            t = Trainer(model_ref, torch.optim.SGD(model_ref.parameters(), lr=0.0), cfg=cfg)
            batch = t._to_device(batch)
            z = model_ref.embed(batch)["z"].detach().cpu().numpy()
            zs.append(z)
            if batch.batch_labels is not None:
                batches.append(batch.batch_labels.detach().cpu().numpy())
    Z_ref = np.vstack(zs)
    labels_ref = np.hstack(batches) if len(batches) > 0 else np.zeros(Z_ref.shape[0], dtype=int)
    mapper = FixedMixtureMapper()
    mapper.fit(torch.from_numpy(Z_ref).float(), torch.from_numpy(labels_ref).long())

    # Load query
    # For simplicity, reuse same cfg/data.path; real use would pass query path
    model_q = model_ref  # in scArches, adapters are added and small finetune is done
    # Optional: attach residual adapters to encoder layers (omitted for brevity)

    adata_q = adata_ref  # replace with query dataset
    loader_q = dataloader(adata_q, cfg, batch_size=cfg.optim.batch_size, shuffle=False)
    Z_query = []
    with torch.no_grad():
        for batch in loader_q:
            from ..train.trainer import Trainer
            t = Trainer(model_q, torch.optim.SGD(model_q.parameters(), lr=0.0), cfg=cfg)
            batch = t._to_device(batch)
            z = model_q.embed(batch)["z"].detach().cpu().numpy()
            Z_query.append(z)
    Z_query = np.vstack(Z_query)
    # Map to reference label space
    preds, probs = mapper.map(torch.from_numpy(Z_query).float(), return_probs=True)
    np.save(os.path.join(output_dir, "Z_ref.npy"), Z_ref)
    np.save(os.path.join(output_dir, "Z_query.npy"), Z_query)
    np.save(os.path.join(output_dir, "labels_ref.npy"), labels_ref)
    np.save(os.path.join(output_dir, "preds_query.npy"), preds.detach().cpu().numpy())
    if probs is not None:
        np.save(os.path.join(output_dir, "probs_query.npy"), probs.detach().cpu().numpy())
    print("Reference mapping artifacts saved to:", output_dir)
    return 0

def main():
    ap = argparse.ArgumentParser("remadom-mapref")
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--ref", type=str, required=True, help="Reference checkpoint")
    ap.add_argument("--query", type=str, required=False, help="Query checkpoint (optional)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    return cli_mapref(args.cfg, args.ref, args.query, args.out)