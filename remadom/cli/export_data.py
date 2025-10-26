from __future__ import annotations
import argparse
import os
import json
from typing import Dict, Any, List, Optional
from remadom.data.export import export_aligned_arrays_blockwise
# You should provide resolve_config, build_registry_from_adata, load_anndata in your codebase.
try:
    from remadom.config.resolve import resolve_config
    from remadom.data.loaders import load_anndata, build_registry_from_adata
    from remadom.data.registries import Registry
except Exception:
    resolve_config = None
    load_anndata = None
    build_registry_from_adata = None
    Registry = None  # type: ignore

def cli_export(cfg_path: str, out_dir: str, formats: Optional[List[str]] = None, chunk_size: int = 8192) -> int:
    if resolve_config is None:
        raise RuntimeError("Config resolver not available")
    cfg = resolve_config([cfg_path])
    os.makedirs(out_dir, exist_ok=True)
    Atrain = load_anndata(cfg.data.source.path)  # type: ignore
    # Registry
    if getattr(cfg.data, "registry_path", None) and os.path.exists(cfg.data.registry_path):
        reg = Registry.load(cfg.data.registry_path)  # type: ignore
    else:
        reg = build_registry_from_adata(Atrain, cfg.data.source.keys)  # type: ignore
        if getattr(cfg.data, "registry_path", None):
            reg.save(cfg.data.registry_path)
    # Format mapping
    fmt_map: Dict[str, str] = {}
    if formats:
        for s in formats:
            if "=" in s:
                m, f = s.split("=", 1)
                fmt_map[m.strip()] = f.strip()
    for m in cfg.data.source.keys.keys():
        fmt_map.setdefault(m, "npy")
    manifest: Dict[str, Any] = {"splits": {}, "registry_path": getattr(cfg.data, "registry_path", "")}
    # Train
    train_meta = {}
    for mod, mk in cfg.data.source.keys.items():
        meta = export_aligned_arrays_blockwise(cfg.data.source.path, {mod: mk}, reg, out_dir, prefix=f"train_{mod}", batch_key=cfg.data.source.batch_key, fmt=fmt_map[mod], chunk_size=chunk_size, mods=[mod])
        train_meta[mod] = meta[mod]
        bpath = meta["batches"]
    manifest["splits"]["train"] = {"modalities": {}, "batches": bpath, "batch_key": cfg.data.source.batch_key}
    for mod, meta in train_meta.items():
        if isinstance(meta, dict) and "shards" in meta:
            manifest["splits"]["train"]["modalities"][mod] = {"type": "csr", **meta}
        else:
            manifest["splits"]["train"]["modalities"][mod] = {"type": "npy_or_npz", "path": meta}
    # Valid
    if getattr(cfg.data, "valid", None) and getattr(cfg.data.valid, "path", None):
        valid_meta = {}
        for mod, mk in cfg.data.valid.keys.items():
            meta = export_aligned_arrays_blockwise(cfg.data.valid.path, {mod: mk}, reg, out_dir, prefix=f"valid_{mod}", batch_key=cfg.data.valid.batch_key, fmt=fmt_map.get(mod, "npy"), chunk_size=chunk_size, mods=[mod])
            valid_meta[mod] = meta[mod]
            vb = meta["batches"]
        manifest["splits"]["valid"] = {"modalities": {}, "batches": vb, "batch_key": cfg.data.valid.batch_key}
        for mod, meta in valid_meta.items():
            if isinstance(meta, dict) and "shards" in meta:
                manifest["splits"]["valid"]["modalities"][mod] = {"type": "csr", **meta}
            else:
                manifest["splits"]["valid"]["modalities"][mod] = {"type": "npy_or_npz", "path": meta}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest to", os.path.join(out_dir, "manifest.json"))
    return 0

def main():
    ap = argparse.ArgumentParser("remadom-export")
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--format", action="append", help="mod=fmt entries, e.g., rna=npy atac=csr adt=npy")
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()
    return cli_export(args.cfg, args.out, args.format, args.chunk)