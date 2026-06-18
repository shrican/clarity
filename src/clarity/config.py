from __future__ import annotations
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.3
    freq_branch: bool = True
    cbam_reduction: int = 16


@dataclass
class DatasetConfig:
    name: str = "wtcherr/unsplash_10k_blur_rand_KS"
    image_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.10
    num_workers: int = 4
    cache_dir: str = "./data/.cache"


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    aux_loss_weight: float = 0.3
    mixup_alpha: float = 0.4
    progressive_sizes: list = field(default_factory=lambda: [128, 192, 224])


@dataclass
class EvalConfig:
    tta: bool = True
    tta_n: int = 5
    threshold: float = 0.5


@dataclass
class InferenceConfig:
    device: str = "mps"
    checkpoint: str = "checkpoints/best.pt"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        if "model" in raw:
            cfg.model = ModelConfig(**raw["model"])
        if "dataset" in raw:
            cfg.dataset = DatasetConfig(**raw["dataset"])
        if "training" in raw:
            cfg.training = TrainingConfig(**raw["training"])
        if "eval" in raw:
            cfg.eval = EvalConfig(**raw["eval"])
        if "inference" in raw:
            cfg.inference = InferenceConfig(**raw["inference"])
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
