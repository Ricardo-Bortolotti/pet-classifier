from pathlib import Path

import pytest

from app.observability.inference_store import InferenceStore


@pytest.fixture
def store(tmp_path: Path) -> InferenceStore:
    return InferenceStore(tmp_path / "inference.db")


def test_save_and_list_recent(store: InferenceStore):
    row_id = store.save(
        filename="cat.jpg",
        prediction="cat",
        probability=0.97,
        latency_ms=42.5,
        model_version="best_model.pth",
        endpoint="predict",
        timestamp="2026-06-09T12:00:00+00:00",
    )

    records = store.list_recent(limit=10)

    assert row_id == 1
    assert store.count() == 1
    assert len(records) == 1
    assert records[0].filename == "cat.jpg"
    assert records[0].prediction == "cat"
    assert records[0].probability == pytest.approx(0.97)
    assert records[0].latency_ms == pytest.approx(42.5)
    assert records[0].model_version == "best_model.pth"
    assert records[0].endpoint == "predict"


def test_list_recent_returns_newest_first(store: InferenceStore):
    store.save(
        filename="first.jpg",
        prediction="cat",
        probability=0.9,
        latency_ms=10.0,
        model_version="v1",
    )
    store.save(
        filename="second.jpg",
        prediction="dog",
        probability=0.8,
        latency_ms=20.0,
        model_version="v1",
    )

    records = store.list_recent(limit=1)

    assert len(records) == 1
    assert records[0].filename == "second.jpg"
    assert store.count() == 2
