import os
import torch
from remadom.config.resolve import resolve_config
from remadom.data.loaders import load_anndata, build_registry_from_adata
from remadom.data.export import export_aligned_arrays_blockwise
from remadom.data.manifest_loader import load_from_manifest
from remadom.config.factories import build_model, build_heads, build_optimizer, apply_head_schedules, get_beta_schedule
from remadom.train.trainer import Trainer
from remadom.utils.serialization import save_yaml
import json

def main():
    cfg = resolve_config(["configs/examples/train_with_valid.yaml"])
    outdir = "runs/smoke_real_manifest"
    os.makedirs(outdir, exist_ok=True)

    # Build registry from train source
    Atrain = load_anndata(cfg.data.source.path)
    reg = build_registry_from_adata(Atrain, cfg.data.source.keys)
    if cfg.data.registry_path:
        reg.save(cfg.data.registry_path)

    # Export RNA/ADT as npy, ATAC (if present) as csr
    fmt_map = {}
    for mod in cfg.data.source.keys.keys():
        fmt_map[mod] = "csr" if mod == "atac" else "npy"

    manifest = {"splits": {}, "registry_path": cfg.data.registry_path or ""}
    # train
    train_mods = {}
    for mod, mk in cfg.data.source.keys.items():
        meta = export_aligned_arrays_blockwise(cfg.data.source.path, {mod: mk}, reg, outdir, prefix=f"train_{mod}", batch_key=cfg.data.source.batch_key, fmt=fmt_map[mod], chunk_size=2048, mods=[mod])
        train_mods[mod] = meta[mod]
        bpath = meta["batches"]
    manifest["splits"]["train"] = {"modalities": {}, "batches": bpath, "batch_key": cfg.data.source.batch_key}
    for mod, meta in train_mods.items():
        if isinstance(meta, dict) and "shards" in meta:
            manifest["splits"]["train"]["modalities"][mod] = {"type": "csr", **meta}
        else:
            manifest["splits"]["train"]["modalities"][mod] = {"type": "npy_or_npz", "path": meta}
    # valid
    if cfg.data.valid and cfg.data.valid.path:
        valid_mods = {}
        for mod, mk in cfg.data.valid.keys.items():
            meta = export_aligned_arrays_blockwise(cfg.data.valid.path, {mod: mk}, reg, outdir, prefix=f"valid_{mod}", batch_key=cfg.data.valid.batch_key, fmt=fmt_map.get(mod, "npy"), chunk_size=2048, mods=[mod])
            valid_mods[mod] = meta[mod]
            vb = meta["batches"]
        manifest["splits"]["valid"] = {"modalities": {}, "batches": vb, "batch_key": cfg.data.valid.batch_key}
        for mod, meta in valid_mods.items():
            if isinstance(meta, dict) and "shards" in meta:
                manifest["splits"]["valid"]["modalities"][mod] = {"type": "csr", **meta}
            else:
                manifest["splits"]["valid"]["modalities"][mod] = {"type": "npy_or_npz", "path": meta}

    man_path = os.path.join(outdir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest:", man_path)

    # Load loaders
    dl_train, dl_valid = load_from_manifest(man_path, batch_size=cfg.optim.batch_size, num_workers=4, pin_memory=torch.cuda.is_available())

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_yaml(cfg.model_dump(), os.path.join(outdir, "config.resolved.yaml"))  # type: ignore[attr-defined]
    model = build_model(cfg).to(device)
    heads = build_heads(cfg)
    head_schedules = apply_head_schedules(heads, cfg)
    beta_init, beta_sched = get_beta_schedule(cfg)
    opt, _ = build_optimizer(cfg, model)
    trainer = Trainer(model, opt, None, heads=heads, cfg=cfg, head_schedules=head_schedules, beta_schedule=beta_sched, beta_init=beta_init, steps_per_epoch=len(dl_train))
    trainer.set_total_steps(len(dl_train))
    logs = trainer.fit(dl_train, dl_valid)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.model_dump()}, os.path.join(outdir, "checkpoint.pt"))  # type: ignore[attr-defined]
    print("Smoke real manifest complete:", outdir)

if __name__ == "__main__":
    main()