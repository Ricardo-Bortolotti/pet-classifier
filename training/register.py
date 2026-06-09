"""CLI to register the champion model in MLflow Model Registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.registry import (
    DEFAULT_EXPERIMENT,
    DEFAULT_MLFLOW_URI,
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_STAGE,
    register_champion_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register the champion model in MLflow Model Registry.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to register (default: read from champion.json).",
    )
    parser.add_argument("--model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_URI)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("app/models/champion.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_champion_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        stage=args.stage,
        experiment_name=args.experiment,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
