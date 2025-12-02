from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: Literal["linear", "cosine"] = "linear"
    start: float = 1.0
    end: float = 1.0
    epochs: Optional[int] = None


class ModalityKeyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    X: str = "X"
    obsm: Optional[str] = None
    var_key: Optional[str] = None
    uns_key: Optional[str] = None


class DataSource(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str
    keys: Dict[str, ModalityKeyConfig] = Field(default_factory=dict)
    batch_key: str = "batch"


class DataConfig(BaseModel):
    source: DataSource
    valid: Optional[DataSource] = None


class EncoderConfig(BaseModel):
    in_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.0


class DecoderConfig(BaseModel):
    out_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    weight: float = 1.0
    weight_schedule: Optional[ScheduleConfig] = None
    library: bool = True
    dispersion: str = "gene"
    params: Dict[str, object] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    latent_bio: int = 16
    encoders: Dict[str, EncoderConfig]
    decoders: Dict[str, DecoderConfig]
    beta: float = 1.0
    beta_schedule: Optional[ScheduleConfig] = None


class MMDConfig(BaseModel):
    enabled: bool = False
    weight: float = 0.1
    bandwidth: float = 1.0
    group_key: str = "batch"
    schedule: Optional[ScheduleConfig] = None


class OTConfig(BaseModel):
    enabled: bool = False
    weight: float = 0.1
    epsilon: float = 0.05
    schedule: Optional[ScheduleConfig] = None
    group_key: str = "batch"


class GWConfig(BaseModel):
    enabled: bool = False
    weight: float = 0.1
    epsilon: float = 0.05
    fused_alpha: float = 0.5
    schedule: Optional[ScheduleConfig] = None
    group_key: str = "batch"


class AlignmentConfig(BaseModel):
    mmd: MMDConfig = MMDConfig()
    ot: OTConfig = OTConfig()
    gw: GWConfig = GWConfig()


class BridgeConfig(BaseModel):
    enabled: bool = False
    method: Literal["mnn", "seeded", "dictionary", "linmap"] = "mnn"
    weight: float = 0.1
    group_key: str = "dataset"
    schedule: Optional[ScheduleConfig] = None
    params: Dict[str, object] = Field(default_factory=dict)
    pairs: Optional[List[Tuple[int, int]]] = None
    normalize: bool = False
    max_edges: Optional[int] = None
    allowed_groups: Optional[List[int]] = None


class AmpConfig(BaseModel):
    enabled: bool = True
    dtype: Literal["fp16", "bf16"] = "bf16"


class EarlyStoppingConfig(BaseModel):
    enabled: bool = False
    monitor: str = "elbo.total"
    mode: Literal["min", "max"] = "min"
    patience: int = 5
    min_delta: float = 1e-4


class SchedulerConfig(BaseModel):
    name: Literal["none", "step", "cosine"] = "none"
    step_size: int = 50
    gamma: float = 0.5


class OptimConfig(BaseModel):
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: Optional[float] = None
    grad_clip_modality: Dict[str, float] = Field(default_factory=dict)
    scheduler: SchedulerConfig = SchedulerConfig()
    amp: AmpConfig = AmpConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


class LoggingConfig(BaseModel):
    run_dir: str = "runs/phase1"
    log_interval: int = 50
    collect_metrics: bool = False

class EvaluationConfig(BaseModel):
    enabled: bool = False
    tasks: List[str] = Field(default_factory=list)
    save_predictions: bool = False


class ExperimentConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    alignment: AlignmentConfig = AlignmentConfig()
    bridge: BridgeConfig = BridgeConfig()
    optim: OptimConfig = OptimConfig()
    logging: LoggingConfig = LoggingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    seed: Optional[int] = None

    def model_dump(self, *args, **kwargs):  # pragma: no cover - convenience for serialization compatibility
        return super().model_dump(*args, **kwargs)
