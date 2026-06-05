import pytest
from pydantic import ValidationError

from app.schemas.prediction import PredictionRequest, PredictionResponse, PredictionResult


def test_prediction_request_defaults():
    request = PredictionRequest()
    assert request.top_k == 3


def test_prediction_request_top_k_bounds():
    with pytest.raises(ValidationError):
        PredictionRequest(top_k=0)


def test_prediction_response():
    response = PredictionResponse(
        model_name="resnet18",
        predictions=[
            PredictionResult(label="cat", confidence=0.95),
            PredictionResult(label="dog", confidence=0.05),
        ],
    )
    assert response.model_name == "resnet18"
    assert len(response.predictions) == 2
