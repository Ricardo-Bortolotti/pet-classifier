from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
        self.model = self.artifact["model"]
        self.class_names: list[str] = self.artifact["class_names"]
        self.model_name: str = self.artifact["model_name"]
        self.image_size: int = self.artifact["image_size"]
        self.transform = get_eval_transforms(self.image_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[PredictionResult]:
        """Classify an image and return top-k predictions."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image=np.array(image))["image"]
        batch = tensor.unsqueeze(0).to(self.device)

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
