from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..align.base import AlignmentHead
from ..core.multimodal import MultimodalManager
from ..typing import Batch


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
        head_schedules: Optional[List[Tuple[AlignmentHead, Optional[Callable[[int], float]]]]] = None,
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

        if hasattr(self.model, "set_beta"):
            self.model.set_beta(self._beta_value)

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
        for epoch in range(epochs):
            self._epoch = epoch
            self._update_schedules(epoch)
            train_stats = self._run_epoch(train_loader, training=True)
            history["train"].append(train_stats)
            if val_loader is not None:
                val_stats = self._run_epoch(val_loader, training=False)
                history["val"].append(val_stats)
                self._maybe_update_best(val_stats)
            if self.sched is not None:
                self.sched.step()
        return history

    # ------------------------------------------------------------------
    def _update_schedules(self, epoch: int) -> None:
        # Alignment head schedules
        for head, schedule in self.head_schedules:
            if schedule is not None and hasattr(head, "set_params"):
                value = float(schedule(epoch))
                head.set_params(epsilon=value)
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
        running: Dict[str, float] = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
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
            for key in ("loss", "recon", "kl"):
                running[key] += float(stats.get(key, 0.0))
            steps += 1
        if steps == 0:
            return {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        return {k: v / steps for k, v in running.items()}

    # ------------------------------------------------------------------
    def _train_step(self, batch: Batch) -> Dict[str, float]:
        self.opt.zero_grad(set_to_none=True)
        autocast_ctx = torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self._amp_dtype)
        with autocast_ctx:
            out = self.manager.masked_elbo(batch, beta=self._beta_value, mod_weights=self._modality_weights)
            loss = out["total"]
            recon = out["recon"]
            kl = out["kl"]
            # Alignment heads operate on fused mean latents (detach handled via autograd)
            z_mean = out["mu"]
            for head, _ in self.head_schedules:
                if head is None:
                    continue
                hloss, _ = head(z_bio=z_mean, groups=getattr(batch, "batch_labels", None))
                loss = loss + hloss

        if self.scaler is not None and self._use_scaler:
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

        return {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_step(self, batch: Batch) -> Dict[str, float]:
        out = self.manager.masked_elbo(batch, beta=self._beta_value, mod_weights=self._modality_weights)
        return {
            "loss": float(out["total"].cpu()),
            "recon": float(out["recon"].cpu()),
            "kl": float(out["kl"].cpu()),
        }

    # ------------------------------------------------------------------
    def _maybe_update_best(self, val_stats: Dict[str, float]) -> None:
        monitor = val_stats.get("loss")
        if monitor is None:
            return
        if self.best_val is None or monitor < self.best_val:
            self.best_val = monitor
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
        }
        torch.save(state, path)

    # ------------------------------------------------------------------
    def _to_device(self, batch: Batch) -> Batch:
        def maybe(t):
            return t.to(self.device) if isinstance(t, torch.Tensor) else t

        batch.x_rna = maybe(batch.x_rna)
        batch.x_atac = maybe(batch.x_atac)
        batch.x_adt = maybe(batch.x_adt)
        batch.has_rna = maybe(batch.has_rna)
        batch.has_atac = maybe(batch.has_atac)
        batch.has_adt = maybe(batch.has_adt)
        batch.libsize_rna = maybe(batch.libsize_rna)
        batch.batch_labels = maybe(batch.batch_labels)
        batch.dataset_labels = maybe(batch.dataset_labels)
        batch.coords = maybe(batch.coords)
        return batch
