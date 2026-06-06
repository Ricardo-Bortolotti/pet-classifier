import torch.nn as nn

from training.config import FreezeStrategy, ModelConfig

EFFICIENTNET_FEATURE_BLOCKS = 7


def get_training_mode(config: ModelConfig) -> str:
    """Return the training strategy label for MLflow comparison."""
    if config.name == "simple_cnn":
        return "from_scratch"

    strategy = config.freeze_strategy
    if strategy == FreezeStrategy.HEAD_ONLY:
        return "transfer_learning"
    if strategy == FreezeStrategy.PARTIAL:
        return "partial_finetuning"
    if strategy == FreezeStrategy.FULL:
        return "full_finetuning"

    return "from_scratch"


def get_experiment_stage(config: ModelConfig) -> str:
    """Map model config to experiment stage for MLflow comparison."""
    mode = get_training_mode(config)
    return {
        "from_scratch": "exp1_baseline",
        "transfer_learning": "exp2_feature_extraction",
        "partial_finetuning": "exp3_partial_finetuning",
        "full_finetuning": "exp4_full_finetuning",
    }[mode]


def describe_freeze_strategy(config: ModelConfig) -> str:
    if config.name != "efficientnet_b0":
        if config.freeze_strategy == FreezeStrategy.HEAD_ONLY:
            return "Backbone frozen, classifier trainable"
        return "All parameters trainable"

    if config.freeze_strategy == FreezeStrategy.HEAD_ONLY:
        return "features[0:7] frozen, classifier trainable"
    if config.freeze_strategy == FreezeStrategy.PARTIAL:
        return "features[0:5] frozen, features[5:7] + classifier trainable"
    return "All parameters trainable (full fine-tuning)"


def apply_freeze_strategy(model: nn.Module, config: ModelConfig) -> None:
    """Apply gradual freezing for EfficientNet-B0 transfer learning."""
    if config.name != "efficientnet_b0":
        if config.freeze_strategy == FreezeStrategy.HEAD_ONLY:
            freeze_backbone(model, config.name)
        return

    for param in model.parameters():
        param.requires_grad = True

    if config.freeze_strategy == FreezeStrategy.HEAD_ONLY:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        return

    if config.freeze_strategy == FreezeStrategy.PARTIAL:
        for module in model.features[:-2]:
            for param in module.parameters():
                param.requires_grad = False
        for module in model.features[-2:]:
            for param in module.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        return

    if config.freeze_strategy == FreezeStrategy.FULL:
        return

    raise ValueError(f"Unknown freeze strategy: {config.freeze_strategy}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze feature extractor layers; only the head remains trainable."""
    if model_name == "efficientnet_b0":
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        return

    if model_name in {"resnet18", "resnet50"}:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        return

    raise ValueError(
        f"Transfer learning is not configured for '{model_name}'. "
        "Supported models: efficientnet_b0, resnet18, resnet50."
    )


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
