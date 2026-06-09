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
from app.inference.predictor import Predictor
from app.schemas.prediction import ExplainResponse, PredictionResponse
from training.models import list_models

MODELS_DIR = Path("app/models")


@lru_cache
def get_predictor() -> Predictor:
    return Predictor(models_dir=MODELS_DIR)


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

        if artifact.get("source") == "registry":
            info["registered_model"] = artifact.get("registered_model", get_registered_model_name())
            info["model_stage"] = artifact.get("model_stage", get_model_stage())
            if artifact.get("registry_version") is not None:
                info["model_version"] = str(artifact["registry_version"])
        elif artifact.get("checkpoint_path") is not None:
            info["checkpoint"] = Path(artifact["checkpoint_path"]).name

    return info


@asynccontextmanager
async def lifespan(_: FastAPI):
    with suppress(FileNotFoundError):
        get_predictor()
    yield


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
    predictions = predictor.predict(image_bytes, top_k=top_k)

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
    try:
        predictions, explained_class, heatmap_base64 = predictor.explain(
            image_bytes,
            top_k=top_k,
            target_label=target_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExplainResponse(
        model_name=predictor.model_name,
        predictions=predictions,
        explained_class=explained_class,
        heatmap_base64=heatmap_base64,
    )
