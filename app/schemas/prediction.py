from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Metadata for an image prediction request."""

    top_k: int = Field(default=3, ge=1, le=10)


class PredictionResult(BaseModel):
    """Single class prediction with confidence score."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """API response for image classification."""

    model_name: str
    predictions: list[PredictionResult]


class ExplainResponse(BaseModel):
    """API response with Grad-CAM visualization."""

    model_name: str
    predictions: list[PredictionResult]
    explained_class: str
    heatmap_base64: str = Field(description="PNG overlay with Grad-CAM heatmap (base64)")


class InferenceLogEntry(BaseModel):
    """Persisted inference event exposed for monitoring."""

    id: int
    timestamp: str
    filename: str
    prediction: str
    probability: float
    latency_ms: float
    model_version: str
    endpoint: str = "predict"


class InferenceMonitoringResponse(BaseModel):
    """Recent inference events for operational monitoring."""

    total: int
    records: list[InferenceLogEntry]
