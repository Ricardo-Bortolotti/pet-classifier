"""Grad-CAM explainability for image classification models."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def resolve_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the convolutional layer used for Grad-CAM per architecture."""
    if model_name == "simple_cnn":
        return model.features[6]
    if model_name in ("resnet18", "resnet50"):
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]

    raise ValueError(
        f"Grad-CAM is not configured for model '{model_name}'. "
        "Supported: simple_cnn, resnet18, resnet50, efficientnet_b0."
    )


class GradCAM:
    """Generate class activation maps from gradients of the target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _save_activations(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self.gradients = grad_output[0]

    def register_hooks(self) -> None:
        self._handles.append(self.target_layer.register_forward_hook(self._save_activations))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a normalized heatmap in [0, 1] for the given class index."""
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        heatmap = cam.squeeze().detach().cpu().numpy()
        heatmap -= heatmap.min()
        heatmap /= heatmap.max() + 1e-8
        return heatmap


@contextmanager
def grad_cam_context(grad_cam: GradCAM) -> Iterator[GradCAM]:
    grad_cam.register_hooks()
    try:
        yield grad_cam
    finally:
        grad_cam.remove_hooks()


def resize_heatmap(heatmap: np.ndarray, height: int, width: int) -> np.ndarray:
    """Upsample heatmap to match the original image size."""
    tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a jet colormap heatmap over the original RGB image."""
    height, width = image.shape[:2]
    heatmap = resize_heatmap(heatmap, height, width)
    colored = cm.jet(heatmap)[:, :, :3]
    normalized = image.astype(np.float32) / 255.0
    blended = (1 - alpha) * normalized + alpha * colored
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def encode_overlay_png(image: Image.Image) -> str:
    """Encode a PIL image as base64 PNG."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_grad_cam_overlay(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_idx: int,
) -> Image.Image:
    """Run Grad-CAM and return the heatmap overlay for one class."""
    target_layer = resolve_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    with grad_cam_context(grad_cam):
        heatmap = grad_cam.generate(input_tensor, class_idx)

    return overlay_heatmap(original_image, heatmap)
