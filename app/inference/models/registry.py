from collections.abc import Callable

import torch.nn as nn
from torchvision import models

from app.inference.config import ModelConfig
from app.inference.models.efficientnet import build_efficientnet_b0
from app.inference.models.simple_cnn import SimpleCNN


def _build_resnet18(config: ModelConfig) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(config.dropout),
        nn.Linear(model.fc.in_features, config.num_classes),
    )
    return model


def _build_resnet50(config: ModelConfig) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if config.pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(config.dropout),
        nn.Linear(model.fc.in_features, config.num_classes),
    )
    return model


def _build_simple_cnn(config: ModelConfig) -> nn.Module:
    return SimpleCNN(
        num_classes=config.num_classes,
        image_size=config.image_size,
    )


MODEL_REGISTRY: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "simple_cnn": _build_simple_cnn,
    "resnet18": _build_resnet18,
    "resnet50": _build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
}


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate a model from the registry."""
    if config.name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{config.name}'. Available: {available}")
    return MODEL_REGISTRY[config.name](config)


def list_models() -> list[str]:
    """Return all registered model architecture names."""
    return sorted(MODEL_REGISTRY)
