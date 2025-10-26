import os
import torch
from remadom.config.resolve import resolve_config
from remadom.config.factories import build_model, build_optimizer
from remadom.align.gw import GWHead
from remadom.data.loaders import dataloader_from_source, build_registry_from_adata, load_anndata
from remadom.train.trainer import Trainer
from remadom.utils.serialization import save_yaml

def main():
    cfg = resolve_config(["configs/examples/unpaired_rna_atac_fgw.yaml"])
    os.makedirs(cfg.logging.run_dir, exist_ok=True)
    # Save resolved config
    save_yaml(cfg.model_dump(), os.path.join(cfg.logging.run_dir, "config.resolved.yaml"))  # type: ignore[attr-defined]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    head = GWHead(weight=cfg.alignment.gw.get("weight", 0.05), epsilon=cfg.alignment.gw.get("epsilon", 1e-3), fused_alpha=cfg.alignment.gw.get("fused_alpha", 0.5))
    opt, sched = build_optimizer(cfg, model)

    # Build registry from source
    adata = load_anndata(cfg.data.source.path)
    reg = build_registry_from_adata(adata, cfg.data.source.keys)

    train_loader = dataloader_from_source(cfg.data.source, cfg, reg=reg, batch_size=cfg.optim.batch_size, shuffle=True)
    trainer = Trainer(model, opt, sched, heads=[head], cfg=cfg)
    logs = trainer.fit(train_loader)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.model_dump()}, os.path.join(cfg.logging.run_dir, "checkpoint.pt"))  # type: ignore[attr-defined]
    print("Finished FGW run.")

if __name__ == "__main__":
    main()