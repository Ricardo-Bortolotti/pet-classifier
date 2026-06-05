from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a single model architecture."""

    name: str
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.2
    image_size: int = 224


@dataclass
class ExperimentConfig:
    """Full experiment configuration tracked by MLflow."""

    experiment_name: str = "petvision-classification"
    run_name: str | None = None
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="simple_cnn", pretrained=False)
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
    mlflow_tracking_uri: str = "mlruns"
    output_dir: Path = Path("app/models")

    @classmethod
    def from_model_name(cls, model_name: str, **overrides) -> "ExperimentConfig":
        """Create an experiment config for a registered model architecture."""
        config = cls(model=ModelConfig(name=model_name))
        for key, value in overrides.items():
            if key == "image_size":
                config.image_size = value
                config.model.image_size = value
            elif hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
        return config
