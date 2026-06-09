from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app, get_inference_store, get_predictor
from app.observability.inference_store import InferenceStore
from app.schemas.prediction import PredictionResult


@pytest.fixture
def inference_store(tmp_path: Path) -> InferenceStore:
    store = InferenceStore(tmp_path / "integration.db")
    get_inference_store.cache_clear()
    get_predictor.cache_clear()
    return store


@pytest.fixture
def client(inference_store: InferenceStore, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    mock_predictor = MagicMock()
    mock_predictor.model_name = "efficientnet_b0"
    mock_predictor.artifact = {
        "model_name": "efficientnet_b0",
        "checkpoint_path": Path("app/models/best_model.pth"),
        "source": "local",
    }
    mock_predictor.predict.return_value = [
        PredictionResult(label="cat", confidence=0.91),
    ]
    mock_predictor.explain.return_value = (
        [PredictionResult(label="cat", confidence=0.91)],
        "cat",
        "aGVsbG8=",
    )

    monkeypatch.setattr("app.api.main.get_predictor", lambda: mock_predictor)
    monkeypatch.setattr("app.api.main.get_inference_store", lambda: inference_store)
    return TestClient(app)


def _sample_jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_predict_persists_inference_record(client: TestClient, inference_store: InferenceStore):
    response = client.post(
        "/predict",
        params={"top_k": 1},
        files={"file": ("exemplo.jpeg", _sample_jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    records = inference_store.list_recent()
    assert len(records) == 1
    assert records[0].filename == "exemplo.jpeg"
    assert records[0].prediction == "cat"
    assert records[0].probability == 0.91
    assert records[0].model_version == "best_model.pth"
    assert records[0].latency_ms >= 0
    assert records[0].endpoint == "predict"


def test_explain_persists_inference_record(client: TestClient, inference_store: InferenceStore):
    response = client.post(
        "/explain",
        params={"top_k": 1},
        files={"file": ("exemplo.jpeg", _sample_jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    records = inference_store.list_recent()
    assert len(records) == 1
    assert records[0].endpoint == "explain"


def test_monitoring_endpoint_returns_persisted_records(client: TestClient):
    client.post(
        "/predict",
        files={"file": ("exemplo.jpeg", _sample_jpeg(), "image/jpeg")},
    )

    response = client.get("/monitoring/inferences?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["records"]) == 1
    assert data["records"][0]["filename"] == "exemplo.jpeg"
    assert data["records"][0]["prediction"] == "cat"
    assert data["records"][0]["probability"] == 0.91
    assert data["records"][0]["model_version"] == "best_model.pth"
    assert "latency_ms" in data["records"][0]
    assert "timestamp" in data["records"][0]
