from __future__ import annotations
import argparse
import os
import json
from typing import Dict, Any, Optional
try:
    import anndata as ad  # type: ignore
except Exception:  # pragma: no cover
    ad = None

def cli_make_zarr_manifest(
    train_zarr: str,
    out_dir: str,
    batch_key: str = "batch",
    modalities: Optional[Dict[str, Dict[str, Any]]] = None,
    valid_zarr: Optional[str] = None,
    registry_path: Optional[str] = None,
) -> int:
    if ad is None:
        raise RuntimeError("anndata is required for Zarr manifest")
    os.makedirs(out_dir, exist_ok=True)
    # Save batches for train (and valid)
    from numpy import save
    Atrain = ad.read_zarr(train_zarr)
    train_batches = os.path.join(out_dir, "train_batches.npy")
    save(train_batches, Atrain.obs[batch_key].values if batch_key in Atrain.obs.columns else 0)
    manifest: Dict[str, Any] = {
        "splits": {
            "train": {
                "modalities": {},
                "batches": train_batches,
                "batch_key": batch_key
            }
        },
        "registry_path": registry_path or ""
    }
    for mod, mk in (modalities or {}).items():
        manifest["splits"]["train"]["modalities"][mod] = {"type": "zarr", "path": train_zarr, "keys": mk}
    if valid_zarr:
        Avalid = ad.read_zarr(valid_zarr)
        valid_batches = os.path.join(out_dir, "valid_batches.npy")
        save(valid_batches, Avalid.obs[batch_key].values if batch_key in Avalid.obs.columns else 0)
        manifest["splits"]["valid"] = {
            "modalities": {},
            "batches": valid_batches,
            "batch_key": batch_key
        }
        for mod, mk in (modalities or {}).items():
            manifest["splits"]["valid"]["modalities"][mod] = {"type": "zarr", "path": valid_zarr, "keys": mk}
    man_path = os.path.join(out_dir, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote Zarr manifest to", man_path)
    return 0

def main():
    ap = argparse.ArgumentParser("remadom-make-zarr-manifest")
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--valid", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch-key", type=str, default="batch")
    ap.add_argument("--registry", type=str, default=None)
    ap.add_argument("--mod", action="append", help="e.g. rna:X=X,var_key=gene_ids or adt:obsm=X_adt,uns_key=adt_names")
    args = ap.parse_args()
    # Parse modalities
    mods: Dict[str, Dict[str, Any]] = {}
    if args.mod:
        for spec in args.mod:
            name, rest = spec.split(":", 1)
            kv = {}
            for pair in rest.split(","):
                k, v = pair.split("=", 1)
                kv[k.strip()] = v.strip()
            mods[name] = kv
    return cli_make_zarr_manifest(args.train, args.out, args.batch_key, mods if mods else None, args.valid, args.registry)