"""MLflow Model Registry operations for champion promotion."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

from app.inference.mlflow_model import PetVisionPyFuncModel
from app.inference.model_registry import get_production_model_uri

DEFAULT_CHAMPION_MANIFEST = Path("app/models/champion.json")
DEFAULT_MLFLOW_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "petvision-classification"
DEFAULT_REGISTERED_MODEL_NAME = "petvision-classifier"
DEFAULT_STAGE = "Production"
REGISTRY_RUN_NAME = "model-registry-promotion"


def load_champion_checkpoint(
    checkpoint_path: Path | None = None,
    manifest_path: Path = DEFAULT_CHAMPION_MANIFEST,
) -> Path:
    """Resolve champion checkpoint path from CLI arg or manifest."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Champion manifest not found: {manifest_path}. "
            "Run `uv run python -m training.compare` first or pass --checkpoint."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    promoted_to = manifest.get("promoted_to")
    if promoted_to:
        path = Path(promoted_to)
        if path.exists():
            return path

    checkpoint_name = manifest.get("checkpoint_path")
    if checkpoint_name:
        path = Path(checkpoint_name)
        if path.exists():
            return path

    raise FileNotFoundError("No valid champion checkpoint found in champion.json.")


def promote_to_production(
    client: MlflowClient,
    *,
    model_name: str,
    version: str,
    stage: str = DEFAULT_STAGE,
) -> None:
    """Transition a registered model version to Production."""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )


def register_champion_model(
    *,
    checkpoint_path: Path | None = None,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    stage: str = DEFAULT_STAGE,
    experiment_name: str = DEFAULT_EXPERIMENT,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_URI,
    manifest_path: Path = DEFAULT_CHAMPION_MANIFEST,
) -> dict[str, str]:
    """Register the champion checkpoint in MLflow Model Registry and promote it."""
    champion_path = load_champion_checkpoint(checkpoint_path, manifest_path)
    checkpoint = torch.load(champion_path, map_location="cpu", weights_only=False)

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    metadata = {
        "model_name": checkpoint["model_name"],
        "class_names": ",".join(checkpoint["class_names"]),
        "image_size": str(checkpoint.get("image_size", 224)),
        "experiment_stage": checkpoint.get("experiment_stage", ""),
        "val_acc": str(manifest.get("val_acc", "")),
        "champion_label": manifest.get("label", champion_path.stem),
    }

    with mlflow.start_run(run_name=REGISTRY_RUN_NAME):
        mlflow.set_tags(
            {
                "registry": "true",
                "champion_label": metadata["champion_label"],
                "registered_model_name": model_name,
            }
        )
        mlflow.log_params(
            {
                "checkpoint_path": str(champion_path),
                "registered_model_name": model_name,
                "target_stage": stage,
            }
        )

        model_info = mlflow.pyfunc.log_model(
            artifact_path="petvision_model",
            python_model=PetVisionPyFuncModel(),
            artifacts={"checkpoint": str(champion_path)},
            registered_model_name=model_name,
            metadata=metadata,
        )

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    version = model_info.registered_model_version
    promote_to_production(client, model_name=model_name, version=version, stage=stage)

    model_uri = get_production_model_uri(model_name, stage)
    result = {
        "registered_model_name": model_name,
        "version": version,
        "stage": stage,
        "model_uri": model_uri,
        "checkpoint_path": str(champion_path),
    }

    print(
        f"\nPetVision AI — Model Registry\n"
        f"Modelo registrado : {model_name}\n"
        f"Versão            : {version}\n"
        f"Stage             : {stage}\n"
        f"URI               : {model_uri}\n"
        f"\nProdução:\n"
        f"  PETVISION_MODEL_SOURCE=registry uv run uvicorn app.api.main:app --reload\n"
    )

    return result
