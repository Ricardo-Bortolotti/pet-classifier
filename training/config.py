from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class FreezeStrategy(StrEnum):
    HEAD_ONLY = "head_only"
    PARTIAL = "partial"
    FULL = "full"


@dataclass
class ModelConfig:
    """Configuration for a single model architecture."""

    name: str
    num_classes: int = 2
    pretrained: bool = True
    freeze_strategy: FreezeStrategy = FreezeStrategy.FULL
    dropout: float = 0.2
    image_size: int = 224

    @property
    def freeze_backbone(self) -> bool:
        return self.freeze_strategy == FreezeStrategy.HEAD_ONLY


@dataclass
class ExperimentConfig:
    """Full experiment configuration tracked by MLflow."""

    experiment_name: str = "petvision-classification"
    run_name: str | None = None
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="simple_cnn",
            pretrained=False,
            freeze_strategy=FreezeStrategy.FULL,
        )
    )
    data_dir: Path = Path("data")
    val_ratio: float = 0.2
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    image_size: int = 224
    num_workers: int = 4
    seed: int = 42
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    output_dir: Path = Path("app/models")

    @classmethod
    def from_model_name(cls, model_name: str, **overrides) -> "ExperimentConfig":
        """Create an experiment config for a registered model architecture."""
        config = cls(model=ModelConfig(name=model_name))

        if model_name == "efficientnet_b0":
            config.model.pretrained = True
            config.model.freeze_strategy = FreezeStrategy.HEAD_ONLY
            config.image_size = 224
            config.model.image_size = 224
            config.learning_rate = 1e-3
        elif model_name == "simple_cnn":
            config.model.pretrained = False
            config.model.freeze_strategy = FreezeStrategy.FULL
            config.image_size = 128
            config.model.image_size = 128
            config.learning_rate = 1e-3

        for key, value in overrides.items():
            if key == "freeze_backbone":
                config.model.freeze_strategy = (
                    FreezeStrategy.HEAD_ONLY if value else FreezeStrategy.FULL
                )
                continue
            if key == "freeze_strategy" and isinstance(value, str):
                value = FreezeStrategy(value)
            if key == "image_size":
                config.image_size = value
                config.model.image_size = value
            elif hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)

        if model_name == "efficientnet_b0" and "learning_rate" not in overrides:
            config.learning_rate = _default_lr_for_strategy(config.model.freeze_strategy)

        return config


def _default_lr_for_strategy(strategy: FreezeStrategy) -> float:
    return {
        FreezeStrategy.HEAD_ONLY: 1e-3,
        FreezeStrategy.PARTIAL: 1e-4,
        FreezeStrategy.FULL: 1e-5,
    }[strategy]
