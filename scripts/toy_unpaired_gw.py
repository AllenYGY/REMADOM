import torch
from remadom.typing import Batch
from remadom.config.schema import ExperimentConfig
from remadom.config.factories import build_model, build_optimizer
from remadom.align.gw import GWHead
from remadom.train.trainer import Trainer

def make_unpaired_batches(n_a=300, n_b=300, d=32, p_rna=128, p_atac=128):
    torch.manual_seed(0)
    z_true = torch.randn(n_a + n_b, d)
    A = torch.randn(d, p_rna) * 0.5
    B = torch.randn(d, p_atac) * 0.5
    X_rna = (z_true[:n_a] @ A).abs()
    X_atac = torch.sigmoid(z_true[n_a:] @ B)
    batch_a = Batch(x_rna=X_rna, has_rna=torch.ones(n_a, dtype=torch.bool), batch_labels=torch.zeros(n_a, dtype=torch.long))
    batch_b = Batch(x_atac=X_atac, has_atac=torch.ones(n_b, dtype=torch.bool), batch_labels=torch.ones(n_b, dtype=torch.long))
    # merge into one Batch with missing modalities
    x_rna = torch.cat([X_rna, torch.zeros(n_b, p_rna)], 0)
    x_atac = torch.cat([torch.zeros(n_a, p_atac), X_atac], 0)
    has_rna = torch.cat([torch.ones(n_a, dtype=torch.bool), torch.zeros(n_b, dtype=torch.bool)], 0)
    has_atac = torch.cat([torch.zeros(n_a, dtype=torch.bool), torch.ones(n_b, dtype=torch.bool)], 0)
    batches = torch.cat([torch.zeros(n_a, dtype=torch.long), torch.ones(n_b, dtype=torch.long)], 0)
    return Batch(x_rna=x_rna, x_atac=x_atac, has_rna=has_rna, has_atac=has_atac, batch_labels=batches)

def main():
    cfg = ExperimentConfig()
    cfg.model.latent_bio = 16
    cfg.model.encoders = {"rna": {"in_dim": 128}, "atac": {"in_dim": 128}}
    cfg.model.decoders = {"rna": {"out_dim": 128}, "atac": {"out_dim": 128}}
    cfg.optim.epochs = 3
    batch = make_unpaired_batches()
    model = build_model(cfg)
    head = GWHead(weight=0.01, epsilon=1e-3)
    opt, sched = build_optimizer(cfg, model)
    trainer = Trainer(model, opt, sched, heads=[head], cfg=cfg)
    loader = [batch for _ in range(5)]
    logs = trainer.fit(loader, None)
    print("Done. last loss:", logs.get("loss"))

if __name__ == "__main__":
    main()