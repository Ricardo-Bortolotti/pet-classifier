import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from app.inference.mlflow_model import PetVisionPyFuncModel
from app.inference.model_registry import (
    ModelRegistry,
    get_production_model_uri,
    resolve_model_source,
)
from training.registry import load_champion_checkpoint, register_champion_model


def test_resolve_model_source_default(monkeypatch):
    monkeypatch.delenv("PETVISION_MODEL_SOURCE", raising=False)
    assert resolve_model_source() == "local"


def test_resolve_model_source_registry(monkeypatch):
    monkeypatch.setenv("PETVISION_MODEL_SOURCE", "registry")
    assert resolve_model_source() == "registry"


def test_get_production_model_uri():
    assert get_production_model_uri("petvision-classifier", "Production") == (
        "models:/petvision-classifier/Production"
    )


def test_load_champion_checkpoint_from_manifest(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_text("x", encoding="utf-8")
    manifest = tmp_path / "champion.json"
    manifest.write_text(
        json.dumps({"promoted_to": str(checkpoint)}),
        encoding="utf-8",
    )

    resolved = load_champion_checkpoint(manifest_path=manifest)

    assert resolved == checkpoint


def test_load_champion_checkpoint_explicit(tmp_path: Path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_text("x", encoding="utf-8")

    resolved = load_champion_checkpoint(checkpoint_path=checkpoint)

    assert resolved == checkpoint


def test_petvision_pyfunc_predict(monkeypatch):
    model = PetVisionPyFuncModel()
    model.artifact = {
        "model": MagicMock(),
        "class_names": ["cat", "dog"],
        "model_name": "simple_cnn",
        "image_size": 128,
    }

    mock_prediction = MagicMock(label="cat", confidence=0.9)

    class FakePredictor:
        def __init__(self, artifact):
            self.class_names = artifact["class_names"]

        def predict(self, image_bytes, top_k=3):
            return [mock_prediction]

    monkeypatch.setattr(
        "app.inference.predictor.Predictor.from_artifact",
        lambda artifact: FakePredictor(artifact),
    )

    result = model.predict(None, pd.DataFrame({"image_bytes": [b"fake-image"]}))

    assert result.iloc[0]["label"] == "cat"
    assert result.iloc[0]["confidence"] == 0.9


@patch("training.registry.promote_to_production")
@patch("training.registry.mlflow")
@patch("training.registry.torch.load")
def test_register_champion_model(mock_torch_load, mock_mlflow, mock_promote, tmp_path: Path):
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_text("x", encoding="utf-8")
    manifest = tmp_path / "champion.json"
    manifest.write_text(
        json.dumps(
            {
                "promoted_to": str(checkpoint),
                "label": "efficientnet_b0_partial",
                "val_acc": 0.91,
            }
        ),
        encoding="utf-8",
    )

    mock_torch_load.return_value = {
        "model_name": "efficientnet_b0",
        "class_names": ["cat", "dog"],
        "image_size": 224,
        "experiment_stage": "exp3_partial_finetuning",
    }
    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.pyfunc.log_model.return_value = MagicMock(registered_model_version="2")

    result = register_champion_model(
        manifest_path=manifest,
        mlflow_tracking_uri="sqlite:///test.db",
    )

    mock_mlflow.pyfunc.log_model.assert_called_once()
    mock_promote.assert_called_once()
    assert result["version"] == "2"
    assert result["registered_model_name"] == "petvision-classifier"


def test_model_registry_local_resolution_uses_champion_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PETVISION_MODEL_SOURCE", "local")

    checkpoint = tmp_path / "best_model.pth"
    manifest = tmp_path / "champion.json"
    manifest.write_text(json.dumps({"promoted_to": str(checkpoint)}), encoding="utf-8")

    registry = ModelRegistry(models_dir=tmp_path)
    resolved = registry._resolve_local_checkpoint()

    assert resolved == checkpoint


@patch("app.inference.model_registry.mlflow.pyfunc.load_model")
@patch("app.inference.model_registry.MlflowClient")
def test_model_registry_load_from_registry(mock_client_cls, mock_load_model, monkeypatch):
    monkeypatch.setenv("PETVISION_MODEL_SOURCE", "registry")

    python_model = PetVisionPyFuncModel()
    python_model.artifact = {
        "model": MagicMock(),
        "class_names": ["cat", "dog"],
        "model_name": "efficientnet_b0",
        "image_size": 224,
    }

    pyfunc_wrapper = MagicMock()
    pyfunc_wrapper._model_impl.python_model = python_model
    mock_load_model.return_value = pyfunc_wrapper

    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [MagicMock(version="3")]
    mock_client_cls.return_value = mock_client

    registry = ModelRegistry()
    artifact = registry.load_from_mlflow_registry()

    assert artifact["source"] == "registry"
    assert artifact["registry_version"] == "3"
    assert artifact["registered_model"] == "petvision-classifier"
