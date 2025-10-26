from __future__ import annotations
from typing import Any, Dict
import json
try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore

def load_yaml(path: str) -> Dict[str, Any]:
    if OmegaConf is None:
        raise RuntimeError("OmegaConf not available; install omegaconf to load YAML")
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore

def save_yaml(obj: Dict[str, Any], path: str) -> None:
    if OmegaConf is None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return
    cfg = OmegaConf.create(obj)
    with open(path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))