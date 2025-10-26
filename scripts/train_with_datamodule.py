import os
import torch
from remadom.config.resolve import resolve_config
from remadom.data.datamodule import CompositeDataModule
from remadom.config.factories import build_model, build_heads, build_optimizer, apply_head_schedules, get_beta_schedule
from remadom.train.trainer import Trainer
from remadom.utils.serialization import save_yaml

def main():
    cfg = resolve_config(["configs/examples/train_with_valid.yaml"])
    os.makedirs(cfg.logging.run_dir, exist_ok=True)
    save_yaml(cfg.model_dump(), os.path.join(cfg.logging.run_dir, "config.resolved.yaml"))  # type: ignore[attr-defined]

    # Choose one:
    # dm = CompositeDataModule.from_config(cfg, num_workers=4)
    dm = CompositeDataModule.from_manifest("runs/mixed_export/manifest.json", batch_size=cfg.optim.batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    heads = build_heads(cfg)
    head_schedules = apply_head_schedules(heads, cfg)
    beta_init, beta_sched = get_beta_schedule(cfg)
    opt, _ = build_optimizer(cfg, model)
    trainer = Trainer(
        model, opt, None, heads=heads, cfg=cfg,
        head_schedules=head_schedules, beta_schedule=beta_sched, beta_init=beta_init,
        steps_per_epoch=dm.steps_per_epoch or 0,
    )
    if dm.steps_per_epoch is not None:
        trainer.set_total_steps(dm.steps_per_epoch)
    logs = trainer.fit(dm.train_loader, dm.valid_loader)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.model_dump()}, os.path.join(cfg.logging.run_dir, "checkpoint.pt"))  # type: ignore[attr-defined]
    print("Training complete.")

if __name__ == "__main__":
    main()