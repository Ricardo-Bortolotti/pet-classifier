import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from app.inference.grad_cam import (
    GradCAM,
    encode_overlay_png,
    generate_grad_cam_overlay,
    grad_cam_context,
    overlay_heatmap,
    resize_heatmap,
    resolve_target_layer,
)
from training.config import ModelConfig
from training.models import build_model


def test_resolve_target_layer_for_supported_models():
    for model_name in ("simple_cnn", "resnet18", "resnet50", "efficientnet_b0"):
        model = build_model(ModelConfig(name=model_name, num_classes=2, pretrained=False))
        layer = resolve_target_layer(model, model_name)
        assert isinstance(layer, torch.nn.Module)


def test_grad_cam_produces_normalized_heatmap():
    config = ModelConfig(name="simple_cnn", num_classes=2, image_size=64, pretrained=False)
    model = build_model(config)
    model.eval()
    target_layer = resolve_target_layer(model, "simple_cnn")
    grad_cam = GradCAM(model, target_layer)
    input_tensor = torch.randn(1, 3, 64, 64)

    with grad_cam_context(grad_cam):
        heatmap = grad_cam.generate(input_tensor, class_idx=0)

    assert heatmap.shape == (16, 16)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_overlay_and_encoding_roundtrip():
    image = np.full((32, 32, 3), 180, dtype=np.uint8)
    heatmap = np.linspace(0, 1, 8 * 8, dtype=np.float32).reshape(8, 8)
    overlay = overlay_heatmap(image, heatmap)
    encoded = encode_overlay_png(overlay)

    decoded = Image.open(BytesIO(base64.b64decode(encoded)))
    assert decoded.size == (32, 32)


def test_resize_heatmap_matches_target_size():
    heatmap = np.ones((7, 7), dtype=np.float32)
    resized = resize_heatmap(heatmap, height=28, width=35)
    assert resized.shape == (28, 35)


def test_generate_grad_cam_overlay_for_efficientnet():
    model = build_model(ModelConfig(name="efficientnet_b0", num_classes=2, pretrained=False))
    model.eval()
    image = np.random.randint(0, 255, size=(96, 96, 3), dtype=np.uint8)
    tensor = torch.randn(1, 3, 224, 224)

    overlay = generate_grad_cam_overlay(model, "efficientnet_b0", tensor, image, class_idx=1)
    assert overlay.size == (96, 96)
