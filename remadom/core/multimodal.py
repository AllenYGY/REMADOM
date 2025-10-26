from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from .vae import MosaicVAE


class MultimodalManager:
    """
    Thin convenience wrapper that exposes encode/fuse/decode helpers around the VAE.
    """

    def __init__(self, model: MosaicVAE) -> None:
        self.model = model

    def encode(self, batch) -> Dict[str, Dict[str, Tensor]]:
        return self.model.encode(batch)

    def fuse(self, enc_outs: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        return self.model.fuse_posteriors(enc_outs)

    def decode(self, z: Tensor, batch) -> Dict[str, Dict[str, Tensor]]:
        return self.model.decode_all(z, batch)

    def masked_elbo(
        self,
        batch,
        *,
        beta: Optional[float] = None,
        mod_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor]:
        return self.model.elbo(batch, beta=beta, mod_weights=mod_weights)

    def impute(
        self,
        batch,
        target_modalities: Optional[List[str]] = None,
        nsamples: int = 1,
    ) -> Dict[str, Dict[str, Tensor]]:
        if nsamples <= 1:
            return self.model.impute(batch, target_modalities=target_modalities)
        outs: Dict[str, List[Dict[str, Tensor]]] = {}
        for _ in range(nsamples):
            sample = self.model.impute(batch, target_modalities=target_modalities, use_latent_sample=True)
            for mod, params in sample.items():
                outs.setdefault(mod, []).append(params)
        merged: Dict[str, Dict[str, Tensor]] = {}
        for mod, lst in outs.items():
            # average parameters where possible
            merged_params: Dict[str, Tensor] = {}
            keys = lst[0].keys()
            for k in keys:
                vals = [item[k] for item in lst if k in item]
                merged_params[k] = torch.stack(vals, 0).mean(0)
            merged[mod] = merged_params
        return merged

    def uncertainty(
        self,
        batch,
        target_modalities: Optional[List[str]] = None,
        nsamples: int = 10,
    ) -> Dict[str, Tensor]:
        enc = self.encode(batch)
        outs: Dict[str, List[Dict[str, Tensor]]] = {}
        for _ in range(nsamples):
            fused = self.fuse(enc)
            z = self.model.reparameterize(fused["mu"], fused["logvar"])
            dec = self.decode(z, batch)
            for m, d in dec.items():
                if target_modalities is not None and m not in target_modalities:
                    continue
                outs.setdefault(m, []).append(d)
        variances: Dict[str, Tensor] = {}
        for mod, lst in outs.items():
            if not lst:
                continue
            if "mu" in lst[0]:
                stack = torch.stack([d["mu"] for d in lst], 0)
                variances[mod] = stack.var(0)
            elif "logits" in lst[0]:
                stack = torch.stack([torch.sigmoid(d["logits"]) for d in lst], 0)
                variances[mod] = stack.var(0)
        return variances
