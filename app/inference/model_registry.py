from pathlib import Path

import torch

from training.config import ModelConfig
from training.models import build_model


class ModelRegistry:
    """Load and cache multiple trained model checkpoints."""

    def __init__(self, models_dir: Path = Path("app/models")) -> None:
        self.models_dir = Path(models_dir)
        self._cache: dict[str, dict] = {}

    def list_checkpoints(self) -> list[Path]:
        """Return all available .pth checkpoints."""
        if not self.models_dir.exists():
            return []
        return sorted(self.models_dir.glob("*.pth"))

    def load(self, checkpoint_path: Path | str | None = None) -> dict:
        """Load a checkpoint and build the corresponding model."""
        if checkpoint_path:
            path = Path(checkpoint_path)
        else:
            candidates = sorted(self.models_dir.glob("*_best.pth"))
            path = candidates[-1] if candidates else self.models_dir / "best_model.pth"

        cache_key = str(path.resolve())
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. "
                "Train a model first with `uv run python -m training.train`."
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_config = ModelConfig(
            name=checkpoint["model_name"],
            num_classes=checkpoint["num_classes"],
            pretrained=False,
            image_size=checkpoint.get("image_size", 224),
        )
        model = build_model(model_config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        artifact = {
            "model": model,
            "class_names": checkpoint["class_names"],
            "model_name": checkpoint["model_name"],
            "image_size": checkpoint.get("image_size", 224),
            "checkpoint_path": path,
        }
        self._cache[cache_key] = artifact
        return artifact
