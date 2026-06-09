from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.inference.grad_cam import encode_overlay_png, generate_grad_cam_overlay
from app.inference.model_registry import ModelRegistry
from app.schemas.prediction import PredictionResult
from training.transforms import get_eval_transforms


class Predictor:
    """Run inference on images using trained checkpoints."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        models_dir: Path = Path("app/models"),
    ) -> None:
        self.registry = ModelRegistry(models_dir=models_dir)
        self.artifact = self.registry.load(checkpoint_path)
        self._init_from_artifact(self.artifact)

    @classmethod
    def from_artifact(cls, artifact: dict) -> "Predictor":
        """Build a predictor from a pre-loaded model artifact."""
        instance = cls.__new__(cls)
        instance.registry = ModelRegistry()
        instance.artifact = artifact
        instance._init_from_artifact(artifact)
        return instance

    def _init_from_artifact(self, artifact: dict) -> None:
        self.model = artifact["model"]
        self.class_names: list[str] = artifact["class_names"]
        self.model_name: str = artifact["model_name"]
        self.image_size: int = artifact["image_size"]
        self.transform = get_eval_transforms(self.image_size)

    @torch.no_grad()
    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[PredictionResult]:
        """Classify an image and return top-k predictions."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image=np.array(image))["image"]
        batch = tensor.unsqueeze(0)

        logits = self.model(batch)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
        top_k = min(top_k, len(self.class_names))
        confidences, indices = torch.topk(probabilities, top_k)

        return [
            PredictionResult(
                label=self.class_names[index],
                confidence=round(confidences[i].item(), 4),
            )
            for i, index in enumerate(indices.tolist())
        ]

    def _prepare_image(self, image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        array = np.array(image)
        tensor = self.transform(image=array)["image"]
        return array, tensor.unsqueeze(0)

    def explain(
        self,
        image_bytes: bytes,
        top_k: int = 3,
        target_label: str | None = None,
    ) -> tuple[list[PredictionResult], str, str]:
        """Classify an image and return predictions plus a Grad-CAM overlay (base64 PNG)."""
        original_image, batch = self._prepare_image(image_bytes)

        self.model.eval()
        logits = self.model(batch)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
        top_k = min(top_k, len(self.class_names))
        confidences, indices = torch.topk(probabilities, top_k)

        predictions = [
            PredictionResult(
                label=self.class_names[index],
                confidence=round(confidences[i].item(), 4),
            )
            for i, index in enumerate(indices.tolist())
        ]

        if target_label is None:
            explained_class = predictions[0].label
            class_idx = indices[0].item()
        else:
            if target_label not in self.class_names:
                raise ValueError(
                    f"Unknown class '{target_label}'. Available: {', '.join(self.class_names)}"
                )
            explained_class = target_label
            class_idx = self.class_names.index(target_label)

        overlay = generate_grad_cam_overlay(
            self.model,
            self.model_name,
            batch,
            original_image,
            class_idx,
        )

        return predictions, explained_class, encode_overlay_png(overlay)
