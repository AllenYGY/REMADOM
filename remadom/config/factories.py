from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR

from ..align.base import AlignmentHead
from ..align.gw import GWHead
from ..align.mmd import MMDHead
from ..align.sinkhorn import SinkhornHead
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
            )
        )
    if cfg.alignment.gw.enabled:
        heads.append(
            GWHead(
                weight=cfg.alignment.gw.weight,
                epsilon=cfg.alignment.gw.epsilon,
                fused_alpha=cfg.alignment.gw.fused_alpha,
            )
        )
    return heads


def apply_head_schedules(heads: Iterable[AlignmentHead], cfg: ExperimentConfig) -> List[Tuple[AlignmentHead, Optional[Callable[[int], float]]]]:
    pairs: List[Tuple[AlignmentHead, Optional[Callable[[int], float]]]] = []
    for head in heads:
        sched_cfg = None
        default = None
        if getattr(head, "name", None) == "sinkhorn" and cfg.alignment.ot.enabled:
            sched_cfg = cfg.alignment.ot.schedule
            default = cfg.alignment.ot.epsilon
        elif getattr(head, "name", None) == "gw" and cfg.alignment.gw.enabled:
            sched_cfg = cfg.alignment.gw.schedule
            default = cfg.alignment.gw.epsilon
        if sched_cfg is not None:
            start, fn = build_schedule(sched_cfg, default if default is not None else 0.05, cfg.optim.epochs)
            if hasattr(head, "set_params"):
                head.set_params(epsilon=start)
            pairs.append((head, fn))
        else:
            pairs.append((head, None))
    return pairs


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
