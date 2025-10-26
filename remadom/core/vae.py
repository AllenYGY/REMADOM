from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn

from ..typing import Batch
from .decoders import DecoderBuildConfig, build_decoder
from .encoders import EncoderBuildConfig, build_encoder
from .fusion import ProductOfExperts
from .losses import kl_gaussian


@dataclass
class MosaicVAEConfig:
    latent_dim: int
    encoder_cfg: Dict[str, EncoderBuildConfig]
    decoder_cfg: Dict[str, DecoderBuildConfig]
    beta: float = 1.0


class MosaicVAE(nn.Module):
    """
    Mosaic-first variational autoencoder with modality-specific encoders/decoders
    and a product-of-experts latent fusion.
    """

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        decoders: Dict[str, nn.Module],
        latent_dim: int,
        fusion: Optional[ProductOfExperts] = None,
        cfg: Optional[object] = None,
    ) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("At least one encoder is required")
        if not decoders:
            raise ValueError("At least one decoder is required")
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.latent_dim = int(latent_dim)
        self.fusion = fusion or ProductOfExperts()
        self.cfg = cfg
        self.register_buffer("prior_mu", torch.zeros(1, self.latent_dim), persistent=False)
        self.register_buffer("prior_logvar", torch.zeros(1, self.latent_dim), persistent=False)
        self._beta_current = float(getattr(cfg.model, "beta", 1.0) if cfg is not None else 1.0)

    @property
    def modalities(self) -> Iterable[str]:
        return self.encoders.keys()

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------
    def encode(self, batch: Batch) -> Dict[str, Dict[str, Tensor]]:
        encodings: Dict[str, Dict[str, Tensor]] = {}
        for mod, encoder in self.encoders.items():
            x = getattr(batch, f"x_{mod}", None)
            if x is None:
                continue
            out = encoder(x)
            mask = getattr(batch, f"has_{mod}", None)
            encodings[mod] = {"mu": out["mu"], "logvar": out["logvar"], "mask": mask}
        return encodings

    def fuse_posteriors(self, encodings: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if not encodings:
            raise ValueError("No modalities present for fusion")
        mus = []
        logvars = []
        masks = []
        names = []
        for mod, out in encodings.items():
            mus.append(out["mu"])
            logvars.append(out["logvar"])
            names.append(mod)
            mask = out.get("mask")
            if mask is None:
                mask = torch.ones(mus[-1].size(0), device=mus[-1].device, dtype=torch.bool)
            masks.append(mask)
        mu_poe, logvar_poe = self.fusion(mus, logvars, modality_names=names, masks=masks)
        return {"mu": mu_poe, "logvar": logvar_poe}

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode_all(self, z: Tensor, batch: Batch) -> Dict[str, Dict[str, Tensor]]:
        outputs: Dict[str, Dict[str, Tensor]] = {}
        for mod, decoder in self.decoders.items():
            if mod == "rna":
                params = decoder(
                    z,
                    libsize=getattr(batch, "libsize_rna", None),
                    batch_labels=getattr(batch, "batch_labels", None),
                )
            else:
                params = decoder(z)
            outputs[mod] = params
        return outputs

    # ------------------------------------------------------------------
    # Losses & utilities
    # ------------------------------------------------------------------
    def set_beta(self, beta: float) -> None:
        self._beta_current = float(beta)

    @staticmethod
    def _subset_params(params: Dict[str, Tensor], mask: Tensor) -> Dict[str, Tensor]:
        if mask is None:
            return params
        mask = mask.bool()
        subset: Dict[str, Tensor] = {}
        for key, value in params.items():
            if isinstance(value, Tensor) and value.shape[0] == mask.shape[0]:
                subset[key] = value[mask]
            else:
                subset[key] = value
        return subset

    def _mod_nll(self, modality: str, x: Tensor, params: Dict[str, Tensor], mask: Optional[Tensor] = None) -> Tensor:
        decoder = self.decoders[modality]
        if mask is not None:
            mask = mask.bool()
            if mask.sum() == 0:
                return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            x = x[mask]
            params = self._subset_params(params, mask)
        return decoder.nll(x, params)

    def elbo(
        self,
        batch: Batch,
        beta: Optional[float] = None,
        mod_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor]:
        encodings = self.encode(batch)
        fused = self.fuse_posteriors(encodings)
        z = self.reparameterize(fused["mu"], fused["logvar"])
        decodings = self.decode_all(z, batch)
        recon_total = torch.tensor(0.0, device=z.device)
        weight_norm = 0.0
        for mod, params in decodings.items():
            x = getattr(batch, f"x_{mod}", None)
            if x is None:
                continue
            has = getattr(batch, f"has_{mod}", None)
            weight = float(mod_weights.get(mod, 1.0) if mod_weights else 1.0)
            recon = self._mod_nll(mod, x, params, mask=has)
            recon_total = recon_total + weight * recon
            weight_norm += weight
        if weight_norm > 0:
            recon_total = recon_total / weight_norm
        kl = kl_gaussian(fused["mu"], fused["logvar"])
        beta_eff = self._beta_current if beta is None else float(beta)
        loss = recon_total + beta_eff * kl
        return {
            "recon": recon_total,
            "kl": kl,
            "beta": torch.tensor(beta_eff, device=z.device),
            "total": loss,
            "z": z,
            "mu": fused["mu"],
            "logvar": fused["logvar"],
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, batch: Batch) -> Dict[str, Tensor]:
        self.eval()
        encodings = self.encode(batch)
        fused = self.fuse_posteriors(encodings)
        return {"z": fused["mu"], "mu": fused["mu"], "logvar": fused["logvar"]}

    @torch.no_grad()
    def impute(
        self,
        batch: Batch,
        target_modalities: Optional[Iterable[str]] = None,
        use_latent_sample: bool = False,
    ) -> Dict[str, Dict[str, Tensor]]:
        self.eval()
        encodings = self.encode(batch)
        fused = self.fuse_posteriors(encodings)
        z = self.reparameterize(fused["mu"], fused["logvar"]) if use_latent_sample else fused["mu"]
        decoded = self.decode_all(z, batch)
        if target_modalities is None:
            return decoded
        return {m: decoded[m] for m in target_modalities if m in decoded}


def build_mosaic_vae_from_config(cfg: MosaicVAEConfig) -> MosaicVAE:
    encoders = {
        mod: build_encoder(mod, enc_cfg, latent_dim=cfg.latent_dim)
        for mod, enc_cfg in cfg.encoder_cfg.items()
    }
    decoders = {
        mod: build_decoder(mod, dec_cfg, latent_dim=cfg.latent_dim)
        for mod, dec_cfg in cfg.decoder_cfg.items()
    }
    return MosaicVAE(encoders, decoders, latent_dim=cfg.latent_dim, fusion=ProductOfExperts())
