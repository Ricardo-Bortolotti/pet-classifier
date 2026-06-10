import os
import time
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference.model_registry import (
    get_model_stage,
    get_registered_model_name,
    resolve_model_source,
)
from app.inference.models import list_models
from app.inference.predictor import Predictor
from app.observability.inference_store import InferenceStore
from app.observability.logging import configure_logging, get_logger
from app.observability.monitoring import record_inference, resolve_model_version
from app.schemas.prediction import (
    ExplainResponse,
    InferenceLogEntry,
    InferenceMonitoringResponse,
    PredictionResponse,
)

MODELS_DIR = Path("app/models")
DEFAULT_INFERENCE_DB = Path("inference_monitoring.db")

api_logger = get_logger("petvision.api")


@lru_cache
def get_predictor() -> Predictor:
    return Predictor(models_dir=MODELS_DIR)


@lru_cache
def get_inference_store() -> InferenceStore:
    db_path = Path(os.environ.get("PETVISION_INFERENCE_DB", DEFAULT_INFERENCE_DB))
    return InferenceStore(db_path)


def _model_health_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "status": "ok",
        "model_loaded": False,
        "model_source": resolve_model_source(),
    }

    with suppress(Exception):
        predictor = get_predictor()
        artifact = predictor.artifact
        info["model_loaded"] = True
        info["model_name"] = predictor.model_name
        info["model_source"] = artifact.get("source", "local")
        info["model_version"] = resolve_model_version(artifact)

        if artifact.get("source") == "registry":
            info["registered_model"] = artifact.get("registered_model", get_registered_model_name())
            info["model_stage"] = artifact.get("model_stage", get_model_stage())
            if artifact.get("registry_version") is not None:
                info["registry_version"] = str(artifact["registry_version"])
        elif artifact.get("checkpoint_path") is not None:
            info["checkpoint"] = Path(artifact["checkpoint_path"]).name

    return info


def _track_inference(
    *,
    filename: str | None,
    predictions: list,
    latency_ms: float,
    artifact: dict[str, Any],
    endpoint: str,
) -> None:
    if not predictions:
        return

    top_prediction = predictions[0]
    record_inference(
        get_inference_store(),
        filename=filename or "unknown",
        prediction=top_prediction.label,
        probability=top_prediction.confidence,
        latency_ms=latency_ms,
        model_version=resolve_model_version(artifact),
        endpoint=endpoint,
    )


def _require_model_at_startup() -> bool:
    return os.environ.get("REQUIRE_MODEL", "").lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(os.environ.get("PETVISION_LOG_LEVEL", "INFO"))
    api_logger.info("api_starting")
    if _require_model_at_startup():
        get_predictor()
        api_logger.info("model_loaded", extra={"model_source": resolve_model_source()})
    else:
        with suppress(FileNotFoundError):
            get_predictor()
            api_logger.info("model_loaded", extra={"model_source": resolve_model_source()})
    yield
    api_logger.info("api_shutdown")


app = FastAPI(
    title="PetVision AI",
    description="Production-grade image classification API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return _model_health_info()


@app.get("/models")
def available_models() -> dict[str, Any]:
    response: dict[str, Any] = {
        "architectures": list_models(),
        "checkpoints": [p.name for p in sorted(MODELS_DIR.glob("*.pth"))],
        "model_source": resolve_model_source(),
    }

    with suppress(Exception):
        predictor = get_predictor()
        artifact = predictor.artifact
        if artifact.get("source") == "registry":
            response["production_model"] = {
                "name": artifact.get("registered_model", get_registered_model_name()),
                "stage": artifact.get("model_stage", get_model_stage()),
                "version": artifact.get("registry_version"),
                "uri": artifact.get("model_uri"),
            }

    return response


@app.get("/monitoring/inferences", response_model=InferenceMonitoringResponse)
def list_inference_logs(limit: int = 50) -> InferenceMonitoringResponse:
    """Return recent persisted inference events for monitoring."""
    limit = max(1, min(limit, 500))
    store = get_inference_store()
    records = [
        InferenceLogEntry(
            id=record.id,
            timestamp=record.timestamp,
            filename=record.filename,
            prediction=record.prediction,
            probability=record.probability,
            latency_ms=record.latency_ms,
            model_version=record.model_version,
            endpoint=record.endpoint,
        )
        for record in store.list_recent(limit=limit)
    ]
    return InferenceMonitoringResponse(total=store.count(), records=records)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File()],
    top_k: int = 3,
) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image_bytes = await file.read()
    started_at = time.perf_counter()
    predictions = predictor.predict(image_bytes, top_k=top_k)
    latency_ms = (time.perf_counter() - started_at) * 1000

    _track_inference(
        filename=file.filename,
        predictions=predictions,
        latency_ms=latency_ms,
        artifact=predictor.artifact,
        endpoint="predict",
    )

    return PredictionResponse(
        model_name=predictor.model_name,
        predictions=predictions,
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(
    file: Annotated[UploadFile, File()],
    top_k: int = 3,
    target_label: str | None = None,
) -> ExplainResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image_bytes = await file.read()
    started_at = time.perf_counter()
    try:
        predictions, explained_class, heatmap_base64 = predictor.explain(
            image_bytes,
            top_k=top_k,
            target_label=target_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - started_at) * 1000

    _track_inference(
        filename=file.filename,
        predictions=predictions,
        latency_ms=latency_ms,
        artifact=predictor.artifact,
        endpoint="explain",
    )

    return ExplainResponse(
        model_name=predictor.model_name,
        predictions=predictions,
        explained_class=explained_class,
        heatmap_base64=heatmap_base64,
    )
