from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference.predictor import Predictor
from app.schemas.prediction import PredictionResponse
from training.models import list_models

MODELS_DIR = Path("app/models")


@lru_cache
def get_predictor() -> Predictor:
    return Predictor(models_dir=MODELS_DIR)


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
def health() -> dict[str, str]:
    checkpoint = MODELS_DIR / "best_model.pth"
    return {
        "status": "ok",
        "model_loaded": str(checkpoint.exists()),
    }


@app.get("/models")
def available_models() -> dict[str, list[str]]:
    return {
        "architectures": list_models(),
        "checkpoints": [p.name for p in sorted(MODELS_DIR.glob("*.pth"))],
    }


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
