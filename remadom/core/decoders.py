from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "RNADecoderZINB",
    "ATACDecoderBernoulli",
    "ADTMixtureDecoder",
    "DecoderBuildConfig",
    "DECODER_REGISTRY",
    "build_decoder",
    "register_decoder",
]


def _make_mlp(latent_dim: int, hidden_dims: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = latent_dim
    for width in hidden_dims:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


@dataclass
class DecoderBuildConfig:
    out_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    weight: float = 1.0
    weight_schedule: Optional[object] = None
    library: bool = True
    dispersion: str = "gene"
    params: Dict[str, object] | None = None


class RNADecoderZINB(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        dispersion: str = "gene",
        library: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            hidden_dims = (256, 256)
        widths = hidden_dims[:-1] if len(hidden_dims) > 1 else ()
        if widths:
            self.hidden = _make_mlp(latent_dim, widths, hidden_dims[-1])
            hidden_out = hidden_dims[-1]
        else:
            self.hidden = nn.Identity()
            hidden_out = latent_dim
        self.mu_head = nn.Linear(hidden_out, out_dim)
        self.logit_pi = nn.Linear(hidden_out, out_dim)
        self.library = library
        if dispersion not in {"gene", "global"}:
            raise ValueError("dispersion must be 'gene' or 'global'")
        if dispersion == "gene":
            self.theta_param = nn.Parameter(torch.ones(out_dim))
            self._theta_shape = "gene"
        else:
            self.theta_param = nn.Parameter(torch.tensor(1.0))
            self._theta_shape = "global"

    def forward(
        self,
        z: Tensor,
        libsize: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        h = self.hidden(z)
        mu = torch.exp(self.mu_head(h))
        if self.library and libsize is not None:
            mu = mu * libsize.unsqueeze(-1)
        if self._theta_shape == "gene":
            theta = self.theta_param.clamp_min(1e-4).expand_as(mu)
        else:
            theta = self.theta_param.clamp_min(1e-4).expand_as(mu)
        pi = torch.sigmoid(self.logit_pi(h))
        return {"mu": mu, "theta": theta, "pi": pi}

    def nll(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        mu = params["mu"].clamp_min(1e-8)
        theta = params["theta"].clamp_min(1e-8)
        pi = params["pi"].clamp(1e-6, 1 - 1e-6)
        lgamma = torch.lgamma
        nb_term = (
            lgamma(theta + x)
            - lgamma(theta)
            - lgamma(x + 1.0)
            + theta * (torch.log(theta) - torch.log(theta + mu))
            + x * (torch.log(mu) - torch.log(theta + mu))
        )
        nb_prob = torch.exp(nb_term)
        zero_case = -torch.log(pi + (1 - pi) * nb_prob + 1e-8)
        nonzero_case = -torch.log(1 - pi + 1e-8) - nb_term
        loss = torch.where(x < 1e-8, zero_case, nonzero_case)
        return loss.mean()


class ATACDecoderBernoulli(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if not hidden_dims:
            hidden_dims = (256, 256)
        self.net = _make_mlp(latent_dim, hidden_dims, out_dim)

    def forward(self, z: Tensor) -> Dict[str, Tensor]:
        logits = self.net(z)
        return {"logits": logits}

    def nll(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        logits = params["logits"]
        return nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="mean")


class ADTMixtureDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (128,),
    ) -> None:
        super().__init__()
        if not hidden_dims:
            hidden_dims = (128,)
        self.net = _make_mlp(latent_dim, hidden_dims, out_dim)
        self.logvar = nn.Parameter(torch.zeros(out_dim))

    def forward(self, z: Tensor) -> Dict[str, Tensor]:
        mu = self.net(z)
        logvar = self.logvar.expand_as(mu)
        return {"mu": mu, "logvar": logvar}

    def nll(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        mu, logvar = params["mu"], params["logvar"]
        return 0.5 * (
            logvar
            + (x - mu) ** 2 / torch.exp(logvar)
            + torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
        ).mean()


DecoderFactory = MutableMapping[str, type[nn.Module]]

DECODER_REGISTRY: DecoderFactory = {
    "rna": RNADecoderZINB,
    "atac": ATACDecoderBernoulli,
    "adt": ADTMixtureDecoder,
}


def register_decoder(name: str, decoder_cls: type[nn.Module]) -> None:
    DECODER_REGISTRY[name] = decoder_cls


def build_decoder(
    name: str,
    cfg: DecoderBuildConfig | Mapping[str, object],
    *,
    latent_dim: int,
) -> nn.Module:
    if name not in DECODER_REGISTRY:
        raise KeyError(f"Unknown decoder '{name}' – available: {sorted(DECODER_REGISTRY)}")
    if isinstance(cfg, Mapping):
        hidden_dims = cfg.get("hidden_dims", (256, 256))
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        dec_cfg = DecoderBuildConfig(
            out_dim=int(cfg.get("out_dim", 0)),
            hidden_dims=tuple(hidden_dims),  # type: ignore[arg-type]
            weight=float(cfg.get("weight", 1.0)),
            weight_schedule=cfg.get("weight_schedule"),
            library=bool(cfg.get("library", True)),
            dispersion=str(cfg.get("dispersion", "gene")),
            params=cfg.get("params"),
        )
    else:
        dec_cfg = cfg
    cls = DECODER_REGISTRY[name]
    kwargs: Dict[str, object] = {}
    if cls is RNADecoderZINB:
        kwargs.update(
            dispersion=dec_cfg.dispersion,
            library=dec_cfg.library,
        )
    if dec_cfg.params:
        kwargs.update(dec_cfg.params)
    return cls(
        latent_dim=latent_dim,
        out_dim=dec_cfg.out_dim,
        hidden_dims=dec_cfg.hidden_dims,
        **kwargs,
    )  # type: ignore[arg-type]
