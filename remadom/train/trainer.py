from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..align.base import AlignmentHead
from ..core.multimodal import MultimodalManager
from ..eval.alignment_metrics import compute_alignment_metrics
from ..typing import Batch

HeadSchedule = Tuple[AlignmentHead, Optional[Tuple[str, Callable[[int], float]]]]


class Trainer:
    """
    Minimal training loop with AMP support, optional alignment heads, modality schedules,
    and checkpoint helpers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        *,
        heads: Optional[List[AlignmentHead]] = None,
        cfg: Optional[object] = None,
        head_schedules: Optional[List[HeadSchedule]] = None,
        beta_schedule: Optional[Callable[[int], float]] = None,
        beta_init: float = 1.0,
        modality_schedules: Optional[Dict[str, Tuple[float, Optional[Callable[[int], float]]]]] = None,
    ) -> None:
        self.model = model
        self.manager = MultimodalManager(model)
        self.opt = optimizer
        self.sched = scheduler
        self.heads = heads or []
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self._epoch = 0
        self.head_schedules = head_schedules or [(h, None) for h in self.heads]
        self._beta_schedule = beta_schedule
        self._beta_value = float(beta_init)
        self._modality_schedules = modality_schedules or {}
        self._modality_weights = {mod: start for mod, (start, _) in self._modality_schedules.items()}

        # AMP configuration
        amp_cfg = getattr(cfg.optim, "amp", None) if cfg is not None and hasattr(cfg, "optim") else None
        self._amp_enabled = bool(getattr(amp_cfg, "enabled", False))
        dtype_str = getattr(amp_cfg, "dtype", "bf16") if amp_cfg is not None else "bf16"
        self._amp_dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16
        if self.device.type != "cuda":
            self._amp_enabled = False
        self._use_scaler = self._amp_enabled and self._amp_dtype == torch.float16 and torch.cuda.is_available()
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler(device_type="cuda", enabled=self._use_scaler)
            except TypeError:  # Older torch versions don't accept device_type
                self.scaler = torch.amp.GradScaler(enabled=self._use_scaler)
        else:  # pragma: no cover - compatibility path for older PyTorch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_scaler)
        if cfg is not None and hasattr(cfg, "optim"):
            clip = getattr(cfg.optim, "grad_clip", None)
            if clip is None or clip <= 0:
                self.grad_clip = None
            else:
                self.grad_clip = float(clip)
        else:
            self.grad_clip = None

        self.best_val: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self._best_metric: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._head_schedule_state: Dict[str, float] = {}
        self._collect_alignment_metrics = False
        if cfg is not None and hasattr(cfg, "logging"):
            self._collect_alignment_metrics = bool(getattr(cfg.logging, "collect_metrics", False))
        self._latest_alignment_metrics: Dict[str, Dict[str, float]] = {}
        self._latest_z: Optional[torch.Tensor] = None
        self._latest_groups: Optional[torch.Tensor] = None
        self._latest_head_loss_trace: Dict[str, float] = {}

        if hasattr(self.model, "set_beta"):
            self.model.set_beta(self._beta_value)
        self._latest_head_details: Dict[str, Dict[str, Any]] = {}
        # Per-modality grad clipping
        self._modality_clip = {}
        if cfg is not None and hasattr(cfg.optim, "grad_clip_modality"):
            self._modality_clip = {k: float(v) for k, v in cfg.optim.grad_clip_modality.items()}

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterable[Batch],
        val_loader: Optional[Iterable[Batch]] = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        history: Dict[str, List[Dict[str, float]]] = {"train": []}
        if val_loader is not None:
            history["val"] = []
        epochs = getattr(self.cfg.optim, "epochs", 1) if self.cfg is not None else 1
        start_epoch = 0
        if hasattr(self, "_resume_state"):
            state_epoch = self._resume_state.get("epoch")
            if isinstance(state_epoch, int):
                self._epoch = state_epoch
                start_epoch = state_epoch + 1
                head_state = self._resume_state.get("head_schedules") or {}
                self._head_schedule_state.update(head_state)
                mod_w = self._resume_state.get("modality_weights")
                if isinstance(mod_w, dict):
                    self._modality_weights.update({k: float(v) for k, v in mod_w.items()})
        for epoch in range(start_epoch, epochs):
            self._epoch = epoch
            self._update_schedules(epoch)
            train_stats = self._run_epoch(train_loader, training=True)
            history["train"].append(train_stats)
            if val_loader is not None:
                val_stats = self.validate(val_loader)
                history["val"].append(val_stats)
                self._maybe_update_best(val_stats)
            if self.sched is not None:
                self.sched.step()
        return history

    # ------------------------------------------------------------------
    def _update_schedules(self, epoch: int) -> None:
        # Alignment head schedules
        for head, schedule in self.head_schedules:
            if schedule is None:
                continue
            param_name, fn = schedule
            value = float(fn(epoch))
            head.set_params(**{param_name: value})
            self._head_schedule_state[f"{head.name}:{param_name}"] = value
        # Beta schedule
        if self._beta_schedule is not None:
            self._beta_value = float(self._beta_schedule(epoch))
            if hasattr(self.model, "set_beta"):
                self.model.set_beta(self._beta_value)
        # Modality weights
        for mod, (start, schedule) in self._modality_schedules.items():
            if schedule is not None:
                self._modality_weights[mod] = float(schedule(epoch))
            else:
                self._modality_weights[mod] = float(start)

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: Iterable[Batch], *, training: bool) -> Dict[str, float]:
        running: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        steps = 0
        if training:
            self.model.train()
        else:
            self.model.eval()
        for batch in loader:
            batch = self._to_device(batch)
            if training:
                stats = self._train_step(batch)
            else:
                stats = self._eval_step(batch)
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    running[key] += float(value)
                    counts[key] += 1
            steps += 1
        if steps == 0:
            return {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        return {k: running[k] / max(1, counts.get(k, steps)) for k in running}

    # ------------------------------------------------------------------
    def _train_step(self, batch: Batch) -> Dict[str, float]:
        self.opt.zero_grad(set_to_none=True)
        if self._amp_enabled:
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                try:
                    autocast_ctx = torch.amp.autocast(
                        device_type="cuda",
                        dtype=self._amp_dtype,
                        enabled=True,
                    )
                except TypeError:
                    autocast_ctx = torch.amp.autocast(dtype=self._amp_dtype, enabled=True)
            else:  # pragma: no cover - compatibility path for older PyTorch
                autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=self._amp_dtype)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            out = self.manager.masked_elbo(batch, beta=self._beta_value, mod_weights=self._modality_weights)
            loss = out["total"]
            recon = out["recon"]
            kl = out["kl"]
            # Alignment heads operate on fused mean latents (detach handled via autograd)
            z_mean = out.get("mu")
            head_total, head_metrics = self._apply_heads(z_mean, batch, phase="train")
            loss = loss + head_total
        if loss.requires_grad:
            if self.scaler is not None and self._use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if self._modality_clip:
                    self.scaler.unscale_(self.opt)
                    self._clip_per_modality()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if self._modality_clip:
                    self._clip_per_modality()
                self.opt.step()
        else:
            # No gradient path (e.g., dummy models in tests); keep optimiser in sync without backward.
            self.opt.step()

        metrics: Dict[str, float] = {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }
        if self.heads:
            metrics["head_total"] = float(head_total.detach().cpu())
        metrics.update(head_metrics)
        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_step(self, batch: Batch) -> Dict[str, float]:
        out = self.manager.masked_elbo(batch, beta=self._beta_value, mod_weights=self._modality_weights)
        loss = out["total"]
        head_total, head_metrics = self._apply_heads(out.get("mu"), batch, phase="eval")
        total = loss + head_total
        metrics: Dict[str, float] = {
            "loss": float(total.detach().cpu()),
            "recon": float(out["recon"].cpu()),
            "kl": float(out["kl"].cpu()),
        }
        if self.heads:
            metrics["head_total"] = float(head_total.detach().cpu())
        metrics.update(head_metrics)
        return metrics

    # ------------------------------------------------------------------
    def validate(self, loader: Iterable[Batch]) -> Dict[str, float]:
        return self._run_epoch(loader, training=False)

    # ------------------------------------------------------------------
    def _apply_heads(self, z_mean: Optional[torch.Tensor], batch: Batch, *, phase: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        if z_mean is None:
            return torch.zeros((), device=self.device, dtype=torch.float32), {}
        if not self.head_schedules:
            metrics: Dict[str, float] = {}
            if self._collect_alignment_metrics:
                group = getattr(batch, "batch_labels", None)
                if group is None:
                    group = getattr(batch, "dataset_labels", None)
                if group is not None:
                    self._latest_z = z_mean.detach().cpu()
                    self._latest_groups = group.detach().cpu()
                    align = compute_alignment_metrics(z_mean.detach(), group.detach())
                    if align:
                        metrics.update({f"align_{k}": v for k, v in align.items()})
                        self._latest_alignment_metrics[phase] = align
            return torch.zeros((), device=self.device, dtype=torch.float32), metrics
        head_sum = torch.zeros((), device=z_mean.device, dtype=z_mean.dtype)
        metrics: Dict[str, float] = {}
        group_lookup: Dict[str, Optional[torch.Tensor]] = {
            "batch": getattr(batch, "batch_labels", None),
            "dataset": getattr(batch, "dataset_labels", None),
            "time": getattr(batch, "time", None),
        }
        aux: Dict[str, Any] = {
            "has": {
                "rna": getattr(batch, "has_rna", None),
                "atac": getattr(batch, "has_atac", None),
                "adt": getattr(batch, "has_adt", None),
            }
        }
        detail_logs: Dict[str, Any] = {}
        for head, _ in self.head_schedules:
            if head is None:
                continue
            group_key = getattr(head, "group_key", "batch")
            groups = group_lookup.get(group_key, group_lookup.get("batch"))
            if groups is None and isinstance(getattr(batch, "meta", None), dict):
                meta_val = batch.meta.get(group_key)  # type: ignore[union-attr]
                if isinstance(meta_val, torch.Tensor):
                    groups = meta_val.to(z_mean.device)
            hloss, hlogs = head(z_bio=z_mean, groups=groups, aux=aux)
            head_sum = head_sum + hloss
            prefix = head.name.replace(".", "_")
            metrics[f"{prefix}_loss"] = float(hloss.detach().cpu())
            self._latest_head_loss_trace[prefix] = float(hloss.detach().cpu())
            if hlogs:
                for key, value in hlogs.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{prefix}_{key}"] = float(value)
                detail_logs[prefix] = hlogs
            if self._collect_alignment_metrics and groups is not None:
                self._latest_z = z_mean.detach().cpu()
                self._latest_groups = groups.detach().cpu()
                align = compute_alignment_metrics(z_mean.detach(), groups.detach())
                if align:
                    align_prefixed = {f"{prefix}_{k}": v for k, v in align.items()}
                    metrics.update(align_prefixed)
                    stored = self._latest_alignment_metrics.get(phase, {})
                    stored.update({f"{prefix}.{k}": v for k, v in align.items()})
                    self._latest_alignment_metrics[phase] = stored
        self._latest_head_details[phase] = detail_logs
        return head_sum, metrics

    # ------------------------------------------------------------------
    def _maybe_update_best(self, val_stats: Dict[str, float]) -> None:
        if isinstance(val_stats, (int, float)):
            monitor = float(val_stats)
        else:
            monitor = val_stats.get("loss")
        if monitor is None:
            return
        if self.best_val is None or monitor < self.best_val:
            self.best_val = monitor
            self._best_metric = monitor
            self._best_epoch = self._epoch
            self.best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "beta": self._beta_value,
            "epoch": self._epoch,
            "head_schedules": self._head_schedule_state,
            "modality_weights": self._modality_weights,
            "latest_head_loss_trace": self._latest_head_loss_trace,
        }
        torch.save(state, path)

    def load_checkpoint(self, state: Dict[str, object]) -> None:
        if "model" in state:
            self.model.load_state_dict(state["model"])  # type: ignore[arg-type]
        if "optimizer" in state and state["optimizer"] is not None:
            self.opt.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        if "scheduler" in state and state["scheduler"] is not None and self.sched is not None:
            self.sched.load_state_dict(state["scheduler"])  # type: ignore[arg-type]
        if "scaler" in state and state["scaler"] is not None:
            try:
                self.scaler.load_state_dict(state["scaler"])  # type: ignore[arg-type]
            except Exception:
                pass
        beta = state.get("beta")
        if beta is not None:
            self._beta_value = float(beta)
            if hasattr(self.model, "set_beta"):
                self.model.set_beta(self._beta_value)
        self._resume_state = state

    # ------------------------------------------------------------------
    def _to_device(self, batch: Batch) -> Batch:
        def maybe(t):
            return t.to(self.device) if isinstance(t, torch.Tensor) else t

        tensor_attrs = (
            "x_rna",
            "x_atac",
            "x_adt",
            "has_rna",
            "has_atac",
            "has_adt",
            "libsize_rna",
            "batch_labels",
            "dataset_labels",
            "time",
            "coords",
        )
        for attr in tensor_attrs:
            if hasattr(batch, attr):
                setattr(batch, attr, maybe(getattr(batch, attr)))
        if hasattr(batch, "modality_masks") and isinstance(batch.modality_masks, dict):
            batch.modality_masks = {
                key: maybe(val)
                for key, val in batch.modality_masks.items()
            }
        return batch

    def _clip_per_modality(self) -> None:
        if not self._modality_clip:
            return
        # Try to clip encoders/decoders separately when keys match modality names
        for mod, clip_val in self._modality_clip.items():
            params = []
            enc = getattr(self.model, "encoders", None)
            dec = getattr(self.model, "decoders", None)
            if enc is not None and mod in enc:
                params += list(enc[mod].parameters())
            if dec is not None and mod in dec:
                params += list(dec[mod].parameters())
            if params:
                torch.nn.utils.clip_grad_norm_(params, clip_val)
