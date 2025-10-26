import os
from remadom.config.resolve import resolve_config
from remadom.data.loaders import load_anndata, build_registry_from_adata
from remadom.data.export import export_aligned_arrays
from remadom.data.npy_dataset import NpyDataset, npy_collate
from torch.utils.data import DataLoader
import torch
from remadom.config.factories import build_model, build_heads, build_optimizer, apply_head_schedules, get_beta_schedule
from remadom.train.trainer import Trainer

def main():
    cfg = resolve_config(["configs/examples/train_with_valid.yaml"])
    os.makedirs(cfg.logging.run_dir, exist_ok=True)
    # Build registry from train
    Atrain = load_anndata(cfg.data.source.path)
    reg = build_registry_from_adata(Atrain, cfg.data.source.keys)
    if cfg.data.registry_path:
        reg.save(cfg.data.registry_path)
    # Export train/valid
    paths_train = export_aligned_arrays(cfg.data.source.path, cfg.data.source.keys, reg, cfg.logging.run_dir, prefix="train", batch_key=cfg.data.source.batch_key, fmt="npy")
    paths_valid = None
    if cfg.data.valid and cfg.data.valid.path:
        paths_valid = export_aligned_arrays(cfg.data.valid.path, cfg.data.valid.keys, reg, cfg.logging.run_dir, prefix="valid", batch_key=cfg.data.valid.batch_key, fmt="npy")
    # Build datasets/loaders
    ds_tr = NpyDataset(paths_train)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.optim.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=npy_collate)
    dl_va = None
    if paths_valid:
        ds_va = NpyDataset(paths_valid)
        dl_va = DataLoader(ds_va, batch_size=cfg.optim.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=npy_collate)
    # Build model/trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    heads = build_heads(cfg)
    head_schedules = apply_head_schedules(heads, cfg)
    beta_init, beta_sched = get_beta_schedule(cfg)
    opt, _ = build_optimizer(cfg, model)
    trainer = Trainer(model, opt, None, heads=heads, cfg=cfg, head_schedules=head_schedules, beta_schedule=beta_sched, beta_init=beta_init, steps_per_epoch=len(dl_tr))
    trainer.set_total_steps(len(dl_tr))
    logs = trainer.fit(dl_tr, dl_va)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.model_dump()}, os.path.join(cfg.logging.run_dir, "checkpoint.pt"))  # type: ignore[attr-defined]
    print("Done.")
if __name__ == "__main__":
    main()