from __future__ import annotations
from typing import Any, Dict
import json

try:  # Prefer OmegaConf when available for full-featured merges / interpolation.
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore

try:  # Fallback to plain YAML if OmegaConf is missing.
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

def load_yaml(path: str) -> Dict[str, Any]:
    if OmegaConf is not None:
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    if yaml is None:
        raise RuntimeError("YAML support not available; install omegaconf or pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data)}")
    return data

def save_yaml(obj: Dict[str, Any], path: str) -> None:
    if OmegaConf is not None:
        cfg = OmegaConf.create(obj)
        with open(path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))
        return
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f)  # type: ignore[arg-type]
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
