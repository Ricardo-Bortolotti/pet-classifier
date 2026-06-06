import torch.nn as nn
from torchvision import models

from training.config import ModelConfig


def build_efficientnet_b0(config: ModelConfig) -> nn.Module:
    """EfficientNet-B0 with optional ImageNet weights and a custom classifier head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.dropout, inplace=True),
        nn.Linear(in_features, config.num_classes),
    )
    return model
