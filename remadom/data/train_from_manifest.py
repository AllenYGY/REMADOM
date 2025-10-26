import os
import torch
from remadom.data.manifest_loader import load_from_manifest
from remadom.config.resolve import resolve_config
from remadom.config.factories import build_model, build_heads, build_optimizer, apply_head_schedules, get_beta_schedule
from remadom.train.trainer import Trainer
from remadom.utils.serialization import save_yaml

def main():
    cfg = resolve_config(["configs/examples/train_with_valid.yaml"])
    man_path = "runs/mixed_export/manifest.json"
    dl_train, dl_valid = load_from_manifest(man_path, batch_size=cfg.optim.batch_size, num_workers=4, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.logging.run_dir, exist_ok=True)
    save_yaml(cfg.model_dump(), os.path.join(cfg.logging.run_dir, "config.resolved.yaml"))  # type: ignore[attr-defined]
    model = build_model(cfg).to(device)
    heads = build_heads(cfg)
    head_schedules = apply_head_schedules(heads, cfg)
    beta_init, beta_sched = get_beta_schedule(cfg)
    opt, _ = build_optimizer(cfg, model)
    trainer = Trainer(model, opt, None, heads=heads, cfg=cfg, head_schedules=head_schedules, beta_schedule=beta_sched, beta_init=beta_init, steps_per_epoch=len(dl_train))
    trainer.set_total_steps(len(dl_train))
    logs = trainer.fit(dl_train, dl_valid)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.model_dump()}, os.path.join(cfg.logging.run_dir, "checkpoint.pt"))  # type: ignore[attr-defined]
    print("Training complete.")
if __name__ == "__main__":
    main()