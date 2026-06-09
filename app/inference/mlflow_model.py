"""MLflow PyFunc wrapper for PetVision classifier models."""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd


class PetVisionPyFuncModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc model that wraps checkpoint loading and inference."""

    def load_context(self, context) -> None:
        from app.inference.model_registry import ModelRegistry

        checkpoint_path = context.artifacts["checkpoint"]
        registry = ModelRegistry()
        self.artifact = registry.load_checkpoint(checkpoint_path)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG002
        from app.inference.predictor import Predictor

        if "image_bytes" not in model_input.columns:
            raise ValueError("model_input must contain an 'image_bytes' column.")

        predictor = Predictor.from_artifact(self.artifact)
        rows: list[dict[str, object]] = []

        for image_bytes in model_input["image_bytes"]:
            predictions = predictor.predict(image_bytes, top_k=min(3, len(predictor.class_names)))
            for prediction in predictions:
                rows.append(
                    {
                        "label": prediction.label,
                        "confidence": prediction.confidence,
                    }
                )

        return pd.DataFrame(rows)
