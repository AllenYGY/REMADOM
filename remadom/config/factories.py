from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR

from ..align.base import AlignmentHead
from ..align.gw import GWHead
from ..align.mmd import MMDHead
from ..align.graph import GraphHead
from ..align.temporal import TemporalHead
from ..align.sinkhorn import SinkhornHead
from ..adapters.bridge_head import BridgeHead, build_bridge_provider
from ..core.vae import MosaicVAE
from ..utils.schedules import (
    build_beta_schedule,
    build_modality_weight_schedules,
    build_schedule,
)
from .schema import ExperimentConfig


def build_model(cfg: ExperimentConfig) -> MosaicVAE:
    latent = cfg.model.latent_bio
    encoders = {
        mod: build_encoder_from_cfg(mod, enc_cfg.model_dump(), latent)
        for mod, enc_cfg in cfg.model.encoders.items()
    }
    decoders = {
        mod: build_decoder_from_cfg(mod, dec_cfg.model_dump(), latent)
        for mod, dec_cfg in cfg.model.decoders.items()
    }
    return MosaicVAE(encoders, decoders, latent_dim=latent, cfg=cfg)


def build_encoder_from_cfg(mod: str, cfg_dict: Dict[str, object], latent: int):
    from ..core.encoders import build_encoder

    return build_encoder(mod, cfg_dict, latent_dim=latent)


def build_decoder_from_cfg(mod: str, cfg_dict: Dict[str, object], latent: int):
    from ..core.decoders import build_decoder

    return build_decoder(mod, cfg_dict, latent_dim=latent)


def build_heads(cfg: ExperimentConfig) -> List[AlignmentHead]:
    heads: List[AlignmentHead] = []
    if cfg.alignment.mmd.enabled:
        heads.append(
            MMDHead(
                weight=cfg.alignment.mmd.weight,
                bandwidth=cfg.alignment.mmd.bandwidth,
                group_key=cfg.alignment.mmd.group_key,
            )
        )
    if cfg.alignment.ot.enabled:
        heads.append(
            SinkhornHead(
                weight=cfg.alignment.ot.weight,
                epsilon=cfg.alignment.ot.epsilon,
                group_key=cfg.alignment.ot.group_key,
            )
        )
    if cfg.alignment.gw.enabled:
        heads.append(
            GWHead(
                weight=cfg.alignment.gw.weight,
                epsilon=cfg.alignment.gw.epsilon,
                fused_alpha=cfg.alignment.gw.fused_alpha,
                group_key=cfg.alignment.gw.group_key,
            )
        )
    if getattr(cfg.alignment, "temporal", None) is not None and cfg.alignment.temporal.enabled:
        heads.append(
            TemporalHead(
                weight=cfg.alignment.temporal.weight,
                group_key=cfg.alignment.temporal.group_key,
            )
        )
    if getattr(cfg, "structure", None) is not None and getattr(cfg.structure, "graph", None) is not None and cfg.structure.graph.enabled:
        heads.append(
            GraphHead(
                weight=cfg.structure.graph.weight,
                k=cfg.structure.graph.k,
                metric=cfg.structure.graph.metric,
                normalized=cfg.structure.graph.normalized,
                lam=cfg.structure.graph.lam,
            )
        )
    if cfg.bridge.enabled:
        bridge_params = dict(cfg.bridge.params)
        normalize = bool(bridge_params.pop("normalize", cfg.bridge.normalize))
        max_edges = bridge_params.pop("max_edges", cfg.bridge.max_edges)
        allowed_groups = bridge_params.pop("allowed_groups", cfg.bridge.allowed_groups)
        provider = build_bridge_provider(cfg.bridge.method, bridge_params)
        heads.append(
            BridgeHead(
                provider=provider,
                weight=cfg.bridge.weight,
                group_key=cfg.bridge.group_key,
                name=f"bridge.{cfg.bridge.method}",
                pairs=cfg.bridge.pairs,
                normalize=normalize,
                max_edges=max_edges,
                allowed_groups=allowed_groups,
            )
        )
    return heads


ScheduleBinding = Optional[Tuple[str, Callable[[int], float]]]


def apply_head_schedules(heads: Iterable[AlignmentHead], cfg: ExperimentConfig) -> List[Tuple[AlignmentHead, ScheduleBinding]]:
    bindings: List[Tuple[AlignmentHead, ScheduleBinding]] = []
    for head in heads:
        name = getattr(head, "name", "")
        schedule_cfg = None
        param_name = None
        default = None
        if name == "sinkhorn" and cfg.alignment.ot.enabled:
            schedule_cfg = cfg.alignment.ot.schedule
            param_name = "epsilon"
            default = cfg.alignment.ot.epsilon
        elif name == "gw" and cfg.alignment.gw.enabled:
            schedule_cfg = cfg.alignment.gw.schedule
            param_name = "epsilon"
            default = cfg.alignment.gw.epsilon
        elif name == "mmd" and cfg.alignment.mmd.enabled:
            schedule_cfg = cfg.alignment.mmd.schedule
            param_name = "weight"
            default = cfg.alignment.mmd.weight
        elif name == "temporal" and getattr(cfg.alignment, "temporal", None) is not None and cfg.alignment.temporal.enabled:
            schedule_cfg = cfg.alignment.temporal.schedule
            param_name = "weight"
            default = cfg.alignment.temporal.weight
        elif name.startswith("bridge") and cfg.bridge.enabled:
            schedule_cfg = cfg.bridge.schedule
            param_name = "weight"
            default = cfg.bridge.weight
        elif name == "graph" and getattr(cfg, "structure", None) is not None and getattr(cfg.structure, "graph", None) is not None and cfg.structure.graph.enabled:
            schedule_cfg = cfg.structure.graph.schedule
            param_name = "weight"
            default = cfg.structure.graph.weight

        if schedule_cfg is None or param_name is None:
            bindings.append((head, None))
            continue

        start, fn = build_schedule(schedule_cfg, default if default is not None else 0.0, cfg.optim.epochs)
        head.set_params(**{param_name: start})
        bindings.append((head, (param_name, fn)))
    return bindings


def build_optimizer(cfg: ExperimentConfig, model: torch.nn.Module) -> Tuple[Optimizer, Optional[LRScheduler]]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler: Optional[LRScheduler] = None
    if cfg.optim.scheduler.name == "step":
        scheduler = StepLR(opt, step_size=cfg.optim.scheduler.step_size, gamma=cfg.optim.scheduler.gamma)
    return opt, scheduler


def get_beta_schedule(cfg: ExperimentConfig) -> Tuple[float, Optional[Callable[[int], float]]]:
    start, fn = build_beta_schedule(cfg.model.beta_schedule, cfg.model.beta, cfg.optim.epochs)
    return float(start), fn


def get_modality_weight_schedules(cfg: ExperimentConfig):
    return build_modality_weight_schedules(cfg.model.decoders, cfg.optim.epochs)
