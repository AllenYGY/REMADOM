import os
import json
import numpy as np
import torch

from remadom.typing import Batch
from remadom.core.vae import MosaicVAE
from remadom.core.encoders import RNAEncoder, ATACEncoder, ADTEncoder
from remadom.core.decoders import RNADecoderZINB, ATACDecoderBernoulli, ADTMixtureDecoder
from remadom.core.fusion import ProductOfExperts
from remadom.train.trainer import Trainer
from remadom.align.mmd import MMDHead
from remadom.eval.evaluator import Evaluator

def make_synth_batch(n=600, d=16, p_rna=200, p_atac=180, p_adt=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    A_rna = torch.randn(d, p_rna, generator=g) * 0.3
    A_atac = torch.randn(d, p_atac, generator=g) * 0.3
    A_adt = torch.randn(d, p_adt, generator=g) * 0.3
    n1 = n // 3; n2 = n // 3; n3 = n - n1 - n2
    z1, z2, z3 = z[:n1], z[n1:n1+n2], z[n1+n2:]
    X_rna = torch.cat([torch.relu(z1 @ A_rna) + 0.1, torch.zeros(n2, p_rna), torch.relu(z3 @ A_rna) + 0.1], 0)
    X_atac = torch.cat([torch.zeros(n1, p_atac), torch.sigmoid(z2 @ A_atac), torch.sigmoid(z3 @ A_atac)], 0)
    X_adt = torch.cat([torch.sigmoid(z1 @ A_adt), torch.zeros(n2, p_adt), torch.sigmoid(z3 @ A_adt)], 0)
    has_rna = torch.cat([torch.ones(n1, dtype=torch.bool), torch.zeros(n2, dtype=torch.bool), torch.ones(n3, dtype=torch.bool)], 0)
    has_atac = torch.cat([torch.zeros(n1, dtype=torch.bool), torch.ones(n2, dtype=torch.bool), torch.ones(n3, dtype=torch.bool)], 0)
    has_adt = torch.cat([torch.ones(n1, dtype=torch.bool), torch.zeros(n2, dtype=torch.bool), torch.ones(n3, dtype=torch.bool)], 0)
    batch_labels = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long), 2*torch.ones(n3, dtype=torch.long)], 0)
    lib = X_rna.sum(1)
    return Batch(
        x_rna=X_rna, x_atac=X_atac, x_adt=X_adt,
        has_rna=has_rna, has_atac=has_atac, has_adt=has_adt,
        libsize_rna=lib, batch_labels=batch_labels
    )

def main():
    outdir = "runs/smoke_synth"
    os.makedirs(outdir, exist_ok=True)

    # Build a minimal model
    d_lat = 16
    enc = {
        "rna": RNAEncoder(in_dim=200, latent_dim=d_lat),
        "atac": ATACEncoder(in_dim=180, latent_dim=d_lat),
        "adt": ADTEncoder(in_dim=20, latent_dim=d_lat),
    }
    dec = {
        "rna": RNADecoderZINB(d_lat, out_dim=200),
        "atac": ATACDecoderBernoulli(d_lat, out_dim=180),
        "adt": ADTMixtureDecoder(d_lat, out_dim=20),
    }
    model = MosaicVAE(enc, dec, latent_dims={"bio": d_lat}, fusion=ProductOfExperts(), cfg=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Data loader from synthetic batch (repeat as iterable)
    batch_full = make_synth_batch()
    def iter_loader(repeats=50, bs=128):
        X = batch_full
        n = X.x_rna.shape[0]
        idx = torch.randperm(n)
        for r in range(repeats):
            for s in range(0, n, bs):
                sl = idx[s:s+bs]
                b = Batch(
                    x_rna=X.x_rna[sl], x_atac=X.x_atac[sl], x_adt=X.x_adt[sl],
                    has_rna=X.has_rna[sl], has_atac=X.has_atac[sl], has_adt=X.has_adt[sl],
                    libsize_rna=X.libsize_rna[sl], batch_labels=X.batch_labels[sl],
                )
                yield b

    heads = [MMDHead(weight=0.05, bandwidth=2.0)]
    from remadom.utils.schedules import build_beta_schedule as _build_beta
    beta0, beta_sched = _build_beta({"kind":"linear","start":0.0,"end":1.0,"epochs":10}, 1.0, 10)
    trainer = Trainer(model, opt, None, heads=heads, cfg=type("C", (), {"optim": type("O", (), {"epochs": 5, "clip": type("CL", (), {"enabled": False})(), "early_stopping": type("ES", (), {"enabled": False})(), "amp": type("A", (), {"enabled": False, "dtype":"fp16"})()})()}), beta_schedule=beta_sched, beta_init=beta0, steps_per_epoch=5)
    trainer.set_total_steps(5)
    logs = trainer.fit(iter_loader(), val_loader=None)
    with open(os.path.join(outdir, "logs.json"), "w") as f:
        json.dump(logs, f, indent=2)

    # Embed and evaluate
    model.eval()
    with torch.no_grad():
        b = batch_full
        for k in b.__dict__.keys():
            v = getattr(b, k)
            if isinstance(v, torch.Tensor):
                setattr(b, k, v.to(device))
        Z = model.embed(b)["z"].detach().cpu().numpy()
        X = b.x_rna.detach().cpu().numpy()
        batches = b.batch_labels.detach().cpu().numpy()

    np.save(os.path.join(outdir, "Z.npy"), Z)
    np.save(os.path.join(outdir, "X.npy"), X)
    np.save(os.path.join(outdir, "batches.npy"), batches)

    ev = Evaluator(["trustworthiness","batch_cls_acc"])
    rep = ev.compute_metrics(torch.tensor(Z, dtype=torch.float32), batches=torch.tensor(batches, dtype=torch.long), aux={"X": torch.tensor(X, dtype=torch.float32)})
    with open(os.path.join(outdir, "report.json"), "w") as f:
        f.write(rep.to_json())
    print("Smoke test complete:", outdir)

if __name__ == "__main__":
    main()