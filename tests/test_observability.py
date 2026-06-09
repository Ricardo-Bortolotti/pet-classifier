import json
import logging
from pathlib import Path

import pytest

from app.observability.inference_store import InferenceStore
from app.observability.logging import JsonFormatter
from app.observability.monitoring import record_inference, resolve_model_version


def test_resolve_model_version_prefers_registry_version():
    artifact = {
        "registry_version": 7,
        "checkpoint_path": Path("app/models/best_model.pth"),
        "model_name": "efficientnet_b0",
    }
    assert resolve_model_version(artifact) == "7"


def test_resolve_model_version_uses_checkpoint_name():
    artifact = {
        "checkpoint_path": Path("app/models/best_model.pth"),
        "model_name": "efficientnet_b0",
    }
    assert resolve_model_version(artifact) == "best_model.pth"


def test_resolve_model_version_falls_back_to_model_name():
    assert resolve_model_version({"model_name": "simple_cnn"}) == "simple_cnn"


def test_json_formatter_emits_structured_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="petvision.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="inference_completed",
        args=(),
        exc_info=None,
    )
    record.image_filename = "cat.jpg"
    record.prediction = "cat"
    record.probability = 0.95

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "inference_completed"
    assert payload["image_filename"] == "cat.jpg"
    assert payload["prediction"] == "cat"
    assert payload["probability"] == 0.95
    assert "timestamp" in payload


def test_record_inference_persists_and_logs(
    store: InferenceStore,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="petvision.inference")

    row_id = record_inference(
        store,
        filename="dog.png",
        prediction="dog",
        probability=0.88,
        latency_ms=33.3,
        model_version="v2",
        endpoint="explain",
    )

    records = store.list_recent()
    assert row_id == 1
    assert records[0].prediction == "dog"
    assert any("inference_completed" in record.message for record in caplog.records)


@pytest.fixture
def store(tmp_path: Path) -> InferenceStore:
    return InferenceStore(tmp_path / "monitoring.db")
