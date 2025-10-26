from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Sequence

from torch import Tensor, nn

__all__ = [
    "BaseEncoder",
    "RNAEncoder",
    "ATACEncoder",
    "ADTEncoder",
    "ENCODER_REGISTRY",
    "EncoderBuildConfig",
    "build_encoder",
    "register_encoder",
]


def _build_mlp(
    in_dim: int, hidden_dims: Sequence[int], dropout: float = 0.0
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for width in hidden_dims:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = width
    return nn.Sequential(*layers)


@dataclass
class EncoderBuildConfig:
    in_dim: int
    latent_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    dropout: float = 0.0


class BaseEncoder(nn.Module):
    """
    Shared MLP encoder that produces modality-specific Gaussian parameters.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            hidden_dims = (256, 256)
        self.backbone = _build_mlp(in_dim, hidden_dims, dropout=dropout)
        last_width = hidden_dims[-1]
        self.mu = nn.Linear(last_width, latent_dim)
        self.logvar = nn.Linear(last_width, latent_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.backbone(x)
        return {"mu": self.mu(h), "logvar": self.logvar(h)}


class RNAEncoder(BaseEncoder):
    pass


class ATACEncoder(BaseEncoder):
    pass


class ADTEncoder(BaseEncoder):
    pass


EncoderFactory = MutableMapping[str, type[BaseEncoder]]

ENCODER_REGISTRY: EncoderFactory = {
    "rna": RNAEncoder,
    "atac": ATACEncoder,
    "adt": ADTEncoder,
}


def register_encoder(name: str, encoder_cls: type[BaseEncoder]) -> None:
    ENCODER_REGISTRY[name] = encoder_cls


def build_encoder(
    name: str,
    cfg: EncoderBuildConfig | Mapping[str, object],
    *,
    latent_dim: int,
) -> BaseEncoder:
    if name not in ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown encoder '{name}' – available: {sorted(ENCODER_REGISTRY)}"
        )
    if isinstance(cfg, Mapping):
        hidden = cfg.get("hidden_dims", (256, 256))
        if isinstance(hidden, int):
            hidden = (hidden,)
        enc_cfg = EncoderBuildConfig(
            in_dim=int(cfg.get("in_dim", 0)),
            latent_dim=latent_dim,
            hidden_dims=tuple(hidden),  # type: ignore[arg-type]
            dropout=float(cfg.get("dropout", 0.0)),
        )
    else:
        enc_cfg = cfg
    return ENCODER_REGISTRY[name](
        in_dim=enc_cfg.in_dim,
        latent_dim=enc_cfg.latent_dim,
        hidden_dims=enc_cfg.hidden_dims,
        dropout=enc_cfg.dropout,
    )
