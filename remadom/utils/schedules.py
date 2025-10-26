from __future__ import annotations

import math
from typing import Callable, Mapping, MutableMapping, Optional


def _to_dict(cfg) -> Optional[dict]:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    if hasattr(cfg, "__dict__"):
        return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
    return None


def linear_schedule(start: float, end: float, epochs: int) -> Callable[[int], float]:
    s, e = float(start), float(end)
    T = max(1, int(epochs))

    def fn(t: int) -> float:
        alpha = min(max(t, 0), T - 1) / max(T - 1, 1)
        return (1 - alpha) * s + alpha * e

    return fn


def cosine_schedule(start: float, end: float, epochs: int) -> Callable[[int], float]:
    s, e = float(start), float(end)
    T = max(1, int(epochs))

    def fn(t: int) -> float:
        x = min(max(t, 0), T)
        cos = (1 + math.cos(math.pi * x / T)) / 2.0
        return e + (s - e) * cos

    return fn


def build_schedule(cfg, default: float, epochs: int):
    data = _to_dict(cfg)
    if not data:
        return float(default), None
    kind = data.get("kind", "linear")
    start = float(data.get("start", default))
    end = float(data.get("end", default))
    T = int(data.get("epochs", epochs))
    if kind == "cosine":
        return start, cosine_schedule(start, end, T)
    return start, linear_schedule(start, end, T)


def build_beta_schedule(cfg, default: float, epochs: int):
    return build_schedule(cfg, default, epochs)


def build_modality_weight_schedules(decoder_cfg: Mapping[str, object], epochs: int):
    schedules: MutableMapping[str, tuple[float, Optional[Callable[[int], float]]]] = {}
    for mod, dc in decoder_cfg.items():
        base_weight = float(getattr(dc, "weight", 1.0))
        sched_cfg = getattr(dc, "weight_schedule", None)
        start = base_weight
        fn = None
        if sched_cfg is not None:
            start, fn = build_schedule(
                sched_cfg,
                default=base_weight,
                epochs=getattr(sched_cfg, "epochs", epochs) or epochs,
            )
        schedules[mod] = (float(start), fn)
    return dict(schedules)


class WarmupCosine:
    """
    Per-step warmup + cosine decay scheduler (to be stepped manually).
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        self.opt = optimizer
        self.base_lr = float(base_lr)
        self.warmup = int(max(0, warmup_steps))
        self.total = int(max(1, total_steps))
        self.min_lr = float(min_lr)
        self._step = 0

    def step(self) -> None:
        self._step += 1
        t = self._step
        if t <= self.warmup and self.warmup > 0:
            lr = self.base_lr * t / self.warmup
        else:
            progress = (t - self.warmup) / max(1, self.total - self.warmup)
            progress = min(max(progress, 0.0), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def get_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])
