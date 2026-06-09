from io import BytesIO
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app, get_predictor
from app.schemas.prediction import ExplainResponse, PredictionResult

client = TestClient(app)


def _sample_jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_source" in data


def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "resnet18" in data["architectures"]
    assert isinstance(data["checkpoints"], list)


def test_predict_endpoint(monkeypatch):
    mock_predictor = MagicMock()
    mock_predictor.model_name = "efficientnet_b0"
    mock_predictor.predict.return_value = [
        PredictionResult(label="cat", confidence=0.95),
        PredictionResult(label="dog", confidence=0.05),
    ]
    monkeypatch.setattr("app.api.main.get_predictor", lambda: mock_predictor)

    response = client.post(
        "/predict",
        params={"top_k": 2},
        files={"file": ("test.jpg", _sample_jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "efficientnet_b0"
    assert data["predictions"][0]["label"] == "cat"
    assert data["predictions"][0]["confidence"] == 0.95
    mock_predictor.predict.assert_called_once()


def test_predict_rejects_non_image():
    get_predictor.cache_clear()
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


def test_explain_endpoint(monkeypatch):
    mock_predictor = MagicMock()
    mock_predictor.model_name = "efficientnet_b0"
    mock_predictor.explain.return_value = (
        [PredictionResult(label="cat", confidence=0.95)],
        "cat",
        "aGVsbG8=",
    )
    monkeypatch.setattr("app.api.main.get_predictor", lambda: mock_predictor)

    response = client.post(
        "/explain",
        params={"top_k": 1},
        files={"file": ("test.jpg", _sample_jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    data = ExplainResponse.model_validate(response.json())
    assert data.model_name == "efficientnet_b0"
    assert data.explained_class == "cat"
    assert data.heatmap_base64 == "aGVsbG8="
    mock_predictor.explain.assert_called_once()


def test_explain_rejects_unknown_target_label(monkeypatch):
    mock_predictor = MagicMock()
    mock_predictor.explain.side_effect = ValueError("Unknown class 'bird'. Available: cat, dog")
    monkeypatch.setattr("app.api.main.get_predictor", lambda: mock_predictor)

    response = client.post(
        "/explain",
        params={"target_label": "bird"},
        files={"file": ("test.jpg", _sample_jpeg(), "image/jpeg")},
    )

    assert response.status_code == 400
    assert "bird" in response.json()["detail"]
