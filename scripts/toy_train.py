import torch
from remadom.typing import Batch
from remadom.config.schema import ExperimentConfig, ModelConfig
from remadom.config.factories import build_model, build_heads, build_optimizer
from remadom.train.trainer import Trainer

def make_toy_batch(n=512, p=100, batches=2):
    torch.manual_seed(0)
    X = torch.randn(n, p).abs() * 2.0  # pseudo counts
    lib = X.sum(1)
    b = torch.randint(0, batches, (n,))
    return Batch(
        x_rna=X,
        has_rna=torch.ones(n, dtype=torch.bool),
        libsize_rna=lib,
        batch_labels=b
    )

def main():
    cfg = ExperimentConfig()
    cfg.model.latent_bio = 16
    cfg.model.encoders = {"rna": {"in_dim": 100}}
    cfg.model.decoders = {"rna": {"out_dim": 100}}
    cfg.optim.epochs = 3
    cfg.alignment.mmd.enabled = True
    cfg.alignment.mmd.weight = 0.1
    batch = make_toy_batch()
    model = build_model(cfg)
    heads = build_heads(cfg)
    opt, sched = build_optimizer(cfg, model)
    trainer = Trainer(model, opt, sched, heads=heads, cfg=cfg)
    loader = [batch for _ in range(10)]
    logs = trainer.fit(loader, None)
    print("Finished. Last loss:", logs.get("loss"))

if __name__ == "__main__":
    main()