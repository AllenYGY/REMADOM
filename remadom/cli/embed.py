from __future__ import annotations
import argparse
import os
import numpy as np
import torch
from ..config.resolve import resolve_config
from ..config.factories import build_model
from ..data.loaders import load_anndata, dataloader

def cli_embed(cfg_path: str, checkpoint: str, output: str) -> int:
    cfg = resolve_config([cfg_path])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    adata = load_anndata(cfg.data.path)
    loader = dataloader(adata, cfg, batch_size=cfg.optim.batch_size, shuffle=False)
    zs = []
    with torch.no_grad():
        for batch in loader:
            # to device
            from ..train.trainer import Trainer
            t = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.0), cfg=cfg)
            batch = t._to_device(batch)  # reuse move helper
            z = model.embed(batch)["z"].detach().cpu().numpy()
            zs.append(z)
    Z = np.vstack(zs)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.save(output, Z)
    print("Saved embeddings to", output)
    return 0

def main():
    ap = argparse.ArgumentParser("remadom-embed")
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    return cli_embed(args.cfg, args.checkpoint, args.output)