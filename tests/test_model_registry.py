import torch

from training.config import ModelConfig
from training.models import build_model, list_models


def test_list_models_returns_registered_architectures():
    models = list_models()
    assert "simple_cnn" in models
    assert "resnet18" in models
    assert "resnet50" in models
    assert "efficientnet_b0" in models


def test_build_model_simple_cnn():
    config = ModelConfig(name="simple_cnn", num_classes=2, image_size=128, pretrained=False)
    model = build_model(config)
    output = model(torch.randn(2, 3, 128, 128))
    assert output.shape == (2, 2)


def test_build_model_resnet18():
    config = ModelConfig(name="resnet18", num_classes=3, pretrained=False)
    model = build_model(config)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 3)


def test_build_model_unknown_raises():
    config = ModelConfig(name="unknown_model")
    try:
        build_model(config)
        raised = False
    except ValueError:
        raised = True
    assert raised
