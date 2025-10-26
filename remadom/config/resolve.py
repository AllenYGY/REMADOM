from __future__ import annotations
from typing import List, Dict, Any
from .schema import ExperimentConfig
from ..utils.serialization import load_yaml
try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore

def resolve_config(paths_or_overrides: List[str]) -> ExperimentConfig:
    """
    Load a YAML config (first element) and apply optional overrides (subsequent key=value pairs)
    using OmegaConf syntax if available. Return a validated ExperimentConfig.
    """
    if not paths_or_overrides:
        return ExperimentConfig()
    data: Dict[str, Any] = load_yaml(paths_or_overrides[0])
    if len(paths_or_overrides) > 1 and OmegaConf is not None:
        base = OmegaConf.create(data)
        overrides = OmegaConf.from_dotlist(paths_or_overrides[1:])
        merged = OmegaConf.merge(base, overrides)
        data = OmegaConf.to_container(merged, resolve=True)  # type: ignore
    return ExperimentConfig(**data)

