import torch

from training.config import FreezeStrategy, ModelConfig
from training.models import build_model
from training.transfer_learning import (
    apply_freeze_strategy,
    count_parameters,
    describe_freeze_strategy,
    get_experiment_stage,
    get_training_mode,
)


def test_get_training_mode_labels():
    assert (
        get_training_mode(
            ModelConfig(
                name="efficientnet_b0",
                pretrained=True,
                freeze_strategy=FreezeStrategy.HEAD_ONLY,
            )
        )
        == "transfer_learning"
    )
    assert (
        get_training_mode(
            ModelConfig(
                name="efficientnet_b0",
                pretrained=True,
                freeze_strategy=FreezeStrategy.PARTIAL,
            )
        )
        == "partial_finetuning"
    )
    assert (
        get_training_mode(
            ModelConfig(
                name="efficientnet_b0",
                pretrained=True,
                freeze_strategy=FreezeStrategy.FULL,
            )
        )
        == "full_finetuning"
    )
    assert get_training_mode(ModelConfig(name="simple_cnn", pretrained=False)) == "from_scratch"


def test_experiment_stage_mapping():
    assert get_experiment_stage(ModelConfig(name="simple_cnn")) == "exp1_baseline"
    assert (
        get_experiment_stage(
            ModelConfig(name="efficientnet_b0", freeze_strategy=FreezeStrategy.HEAD_ONLY)
        )
        == "exp2_feature_extraction"
    )
    assert (
        get_experiment_stage(
            ModelConfig(name="efficientnet_b0", freeze_strategy=FreezeStrategy.PARTIAL)
        )
        == "exp3_partial_finetuning"
    )
    assert (
        get_experiment_stage(
            ModelConfig(name="efficientnet_b0", freeze_strategy=FreezeStrategy.FULL)
        )
        == "exp4_full_finetuning"
    )


def test_freeze_efficientnet_head_only():
    config = ModelConfig(
        name="efficientnet_b0",
        num_classes=2,
        pretrained=False,
        freeze_strategy=FreezeStrategy.HEAD_ONLY,
    )
    model = build_model(config)
    apply_freeze_strategy(model, config)

    assert all(not param.requires_grad for param in model.features.parameters())
    assert any(param.requires_grad for param in model.classifier.parameters())


def test_freeze_efficientnet_partial():
    config = ModelConfig(
        name="efficientnet_b0",
        num_classes=2,
        pretrained=False,
        freeze_strategy=FreezeStrategy.PARTIAL,
    )
    model = build_model(config)
    apply_freeze_strategy(model, config)

    head_only_trainable = count_parameters(model, trainable_only=True)

    for module in model.features[:-2]:
        assert all(not param.requires_grad for param in module.parameters())
    for module in model.features[-2:]:
        assert any(param.requires_grad for param in module.parameters())
    assert any(param.requires_grad for param in model.classifier.parameters())

    full_config = ModelConfig(
        name="efficientnet_b0",
        num_classes=2,
        pretrained=False,
        freeze_strategy=FreezeStrategy.FULL,
    )
    full_model = build_model(full_config)
    apply_freeze_strategy(full_model, full_config)
    full_trainable = count_parameters(full_model, trainable_only=True)

    assert head_only_trainable < full_trainable


def test_freeze_efficientnet_full():
    config = ModelConfig(
        name="efficientnet_b0",
        num_classes=2,
        pretrained=False,
        freeze_strategy=FreezeStrategy.FULL,
    )
    model = build_model(config)
    apply_freeze_strategy(model, config)

    assert all(param.requires_grad for param in model.parameters())


def test_describe_freeze_strategy():
    partial = ModelConfig(name="efficientnet_b0", freeze_strategy=FreezeStrategy.PARTIAL)
    assert "features[0:5]" in describe_freeze_strategy(partial)


def test_efficientnet_b0_forward_pretrained_head():
    config = ModelConfig(
        name="efficientnet_b0",
        num_classes=2,
        pretrained=False,
        image_size=224,
    )
    model = build_model(config)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 2)
