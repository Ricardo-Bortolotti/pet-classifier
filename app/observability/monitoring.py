"""Inference monitoring: structured logs + persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.observability.inference_store import InferenceStore
from app.observability.logging import get_logger, log_event

logger = get_logger("petvision.inference")


def resolve_model_version(artifact: dict[str, Any]) -> str:
    """Build a stable model version label from a loaded artifact."""
    if artifact.get("registry_version") is not None:
        return str(artifact["registry_version"])

    checkpoint_path = artifact.get("checkpoint_path")
    if checkpoint_path is not None:
        return Path(checkpoint_path).name

    return str(artifact.get("model_name", "unknown"))


def record_inference(
    store: InferenceStore,
    *,
    filename: str,
    prediction: str,
    probability: float,
    latency_ms: float,
    model_version: str,
    endpoint: str = "predict",
) -> int:
    """Log and persist a single inference event."""
    row_id = store.save(
        filename=filename,
        prediction=prediction,
        probability=probability,
        latency_ms=latency_ms,
        model_version=model_version,
        endpoint=endpoint,
    )

    log_event(
        logger,
        "inference_completed",
        inference_id=row_id,
        image_filename=filename,
        prediction=prediction,
        probability=probability,
        latency_ms=round(latency_ms, 2),
        model_version=model_version,
        endpoint=endpoint,
    )
    return row_id
