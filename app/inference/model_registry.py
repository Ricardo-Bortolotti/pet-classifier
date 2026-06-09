import json
import os
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

from training.config import ModelConfig
from training.models import build_model

DEFAULT_MODELS_DIR = Path("app/models")
DEFAULT_REGISTERED_MODEL_NAME = "petvision-classifier"
DEFAULT_MODEL_STAGE = "Production"
DEFAULT_MLFLOW_URI = "sqlite:///mlflow.db"


def resolve_model_source() -> str:
    """Return configured model source: local or registry."""
    return os.environ.get("PETVISION_MODEL_SOURCE", "local").lower()


def get_registered_model_name() -> str:
    return os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME)


def get_model_stage() -> str:
    return os.environ.get("MLFLOW_MODEL_STAGE", DEFAULT_MODEL_STAGE)


def get_mlflow_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)


def get_production_model_uri(
    name: str | None = None,
    stage: str | None = None,
) -> str:
    """Build the models:/ URI for a registered model stage."""
    model_name = name or get_registered_model_name()
    model_stage = stage or get_model_stage()
    return f"models:/{model_name}/{model_stage}"


class ModelRegistry:
    """Load and cache trained model checkpoints locally or from MLflow Registry."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self._cache: dict[str, dict] = {}

    def list_checkpoints(self) -> list[Path]:
        """Return all available .pth checkpoints."""
        if not self.models_dir.exists():
            return []
        return sorted(self.models_dir.glob("*.pth"))

    def load(self, checkpoint_path: Path | str | None = None) -> dict:
        """Load a model from registry or local checkpoints."""
        if checkpoint_path is not None:
            return self.load_checkpoint(checkpoint_path)

        if resolve_model_source() == "registry":
            return self.load_from_mlflow_registry()

        return self.load_checkpoint(self._resolve_local_checkpoint())

    def load_checkpoint(self, checkpoint_path: Path | str) -> dict:
        """Load a model artifact from a local checkpoint file."""
        path = Path(checkpoint_path)
        cache_key = str(path.resolve())
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. "
                "Train a model first with `uv run python -m training.train`."
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        artifact = self._build_artifact_from_checkpoint(checkpoint, path)
        artifact["source"] = "local"
        self._cache[cache_key] = artifact
        return artifact

    def load_from_mlflow_registry(
        self,
        name: str | None = None,
        stage: str | None = None,
        tracking_uri: str | None = None,
    ) -> dict:
        """Load the production model from MLflow Model Registry."""
        model_name = name or get_registered_model_name()
        model_stage = stage or get_model_stage()
        uri = get_production_model_uri(model_name, model_stage)
        cache_key = f"registry:{uri}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        mlflow.set_tracking_uri(tracking_uri or get_mlflow_tracking_uri())
        pyfunc_model = mlflow.pyfunc.load_model(uri)
        python_model = pyfunc_model._model_impl.python_model  # noqa: SLF001
        artifact = dict(python_model.artifact)
        artifact["source"] = "registry"
        artifact["registered_model"] = model_name
        artifact["model_stage"] = model_stage

        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        if versions:
            artifact["registry_version"] = versions[0].version

        artifact["model_uri"] = uri
        self._cache[cache_key] = artifact
        return artifact

    def _resolve_local_checkpoint(self) -> Path:
        manifest_path = self.models_dir / "champion.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            promoted_to = Path(manifest.get("promoted_to", self.models_dir / "best_model.pth"))
            if promoted_to.exists():
                return promoted_to

        best_model = self.models_dir / "best_model.pth"
        if best_model.exists():
            return best_model

        candidates = sorted(self.models_dir.glob("*_best.pth"))
        if candidates:
            return candidates[-1]

        return best_model

    def _build_artifact_from_checkpoint(self, checkpoint: dict, path: Path) -> dict:
        model_config = ModelConfig(
            name=checkpoint["model_name"],
            num_classes=checkpoint["num_classes"],
            pretrained=False,
            image_size=checkpoint.get("image_size", 224),
        )
        model = build_model(model_config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return {
            "model": model,
            "class_names": checkpoint["class_names"],
            "model_name": checkpoint["model_name"],
            "image_size": checkpoint.get("image_size", 224),
            "checkpoint_path": path,
            "experiment_stage": checkpoint.get("experiment_stage"),
        }
