"""Final model comparison and champion selection."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import torch

from app.inference.model_registry import ModelRegistry
from training.dataset import create_dataloaders
from training.evaluate import collect_predictions, evaluate_full
from training.plots import generate_comparison_plots

DEFAULT_MODELS_DIR = Path("app/models")
DEFAULT_CHAMPION_PATH = DEFAULT_MODELS_DIR / "best_model.pth"
DEFAULT_CHAMPION_MANIFEST = DEFAULT_MODELS_DIR / "champion.json"
DEFAULT_MLFLOW_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "petvision-classification"
PARENT_RUN_NAME = "final-model-comparison"


@dataclass
class ModelEvaluationResult:
    """Evaluation outcome for a single checkpoint."""

    label: str
    checkpoint_path: Path
    experiment_stage: str | None
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_y_true: list[int] = field(default_factory=list)
    test_y_pred: list[int] = field(default_factory=list)
    test_y_proba: list[float] = field(default_factory=list)

    @property
    def val_acc(self) -> float:
        return self.val_metrics["accuracy"]


def discover_checkpoints(models_dir: Path = DEFAULT_MODELS_DIR) -> list[Path]:
    """Return all top-level *_best.pth checkpoints."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*_best.pth"))


def checkpoint_label(checkpoint_path: Path) -> str:
    """Build a friendly label from a checkpoint filename."""
    name = checkpoint_path.name
    if name.endswith("_best.pth"):
        return name[: -len("_best.pth")]
    return checkpoint_path.stem


def select_champion(results: list[ModelEvaluationResult]) -> ModelEvaluationResult:
    """Select the champion model by highest val_acc."""
    if not results:
        raise ValueError("No evaluation results available to select a champion.")

    return max(
        results,
        key=lambda result: (
            result.val_acc,
            result.test_metrics["accuracy"],
            result.test_metrics["f1_score"],
        ),
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    data_dir: Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    models_dir: Path,
) -> ModelEvaluationResult:
    """Load and evaluate a checkpoint on validation and test splits."""
    registry = ModelRegistry(models_dir=models_dir)
    artifact = registry.load(checkpoint_path)
    model = artifact["model"]
    image_size = artifact["image_size"]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment_stage = checkpoint.get("experiment_stage")

    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
    )

    val_metrics = evaluate_full(model, loaders.val)
    test_metrics = evaluate_full(model, loaders.test)
    test_y_true, test_y_pred, test_y_proba = collect_predictions(model, loaders.test)

    return ModelEvaluationResult(
        label=checkpoint_label(checkpoint_path),
        checkpoint_path=checkpoint_path,
        experiment_stage=experiment_stage,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_y_true=test_y_true,
        test_y_pred=test_y_pred,
        test_y_proba=test_y_proba,
    )


def serialize_results(results: list[ModelEvaluationResult]) -> list[dict]:
    """Convert evaluation results to JSON-serializable records."""
    records: list[dict] = []
    for result in results:
        records.append(
            {
                "label": result.label,
                "checkpoint_path": str(result.checkpoint_path),
                "experiment_stage": result.experiment_stage,
                "val_acc": result.val_acc,
                "val_metrics": result.val_metrics,
                "test_metrics": result.test_metrics,
            }
        )
    return records


def promote_champion(
    champion: ModelEvaluationResult,
    *,
    champion_path: Path = DEFAULT_CHAMPION_PATH,
    manifest_path: Path = DEFAULT_CHAMPION_MANIFEST,
) -> Path:
    """Copy champion checkpoint and write manifest JSON."""
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(champion.checkpoint_path, champion_path)

    manifest = {
        "checkpoint_name": champion.checkpoint_path.name,
        "checkpoint_path": str(champion.checkpoint_path),
        "label": champion.label,
        "experiment_stage": champion.experiment_stage,
        "selection_criterion": "highest val_acc",
        "val_acc": champion.val_acc,
        "test_metrics": champion.test_metrics,
        "promoted_to": str(champion_path),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return champion_path


def run_final_comparison(
    *,
    data_dir: Path = Path("data"),
    models_dir: Path = DEFAULT_MODELS_DIR,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    experiment_name: str = DEFAULT_EXPERIMENT,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_URI,
    reports_dir: Path = Path("reports/final_comparison"),
) -> ModelEvaluationResult:
    """Compare all checkpoints, log to MLflow and promote the champion."""
    checkpoints = discover_checkpoints(models_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in '{models_dir}'. Train models before running comparison."
        )

    results = [
        evaluate_checkpoint(
            checkpoint_path,
            data_dir=data_dir,
            batch_size=batch_size,
            val_ratio=val_ratio,
            seed=seed,
            num_workers=num_workers,
            models_dir=models_dir,
        )
        for checkpoint_path in checkpoints
    ]
    champion = select_champion(results)
    champion_path = promote_champion(champion)

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = reports_dir / "comparison_results.json"
    comparison_json.write_text(
        json.dumps(serialize_results(results), indent=2) + "\n",
        encoding="utf-8",
    )

    plot_paths = generate_comparison_plots(
        results,
        reports_dir,
        champion_label=champion.label,
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=PARENT_RUN_NAME):
        mlflow.set_tags(
            {
                "final_evaluation": "true",
                "champion": champion.label,
                "champion_val_acc": f"{champion.val_acc:.6f}",
            }
        )
        mlflow.log_params(
            {
                "selection_criterion": "highest val_acc",
                "champion_checkpoint": str(champion.checkpoint_path),
                "champion_promoted_to": str(champion_path),
                "num_models_compared": len(results),
            }
        )

        for result in results:
            prefix = result.label
            mlflow.log_metrics(
                {
                    f"{prefix}_val_acc": result.val_acc,
                    f"{prefix}_test_accuracy": result.test_metrics["accuracy"],
                    f"{prefix}_test_precision": result.test_metrics["precision"],
                    f"{prefix}_test_recall": result.test_metrics["recall"],
                    f"{prefix}_test_f1_score": result.test_metrics["f1_score"],
                    f"{prefix}_test_roc_auc": result.test_metrics["roc_auc"],
                }
            )

        mlflow.log_artifact(str(comparison_json))
        mlflow.log_artifact(str(DEFAULT_CHAMPION_MANIFEST))
        for plot_path in plot_paths.values():
            mlflow.log_artifact(str(plot_path))

    print(
        f"\nPetVision AI — Final Evaluation\n"
        f"Models compared    : {len(results)}\n"
        f"Champion           : {champion.label}\n"
        f"val_acc            : {champion.val_acc:.4f}\n"
        f"test accuracy      : {champion.test_metrics['accuracy']:.4f}\n"
        f"Promoted to        : {champion_path}\n"
        f"Reports            : {reports_dir}\n"
    )

    return champion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained models and select champion.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_URI)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/final_comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_final_comparison(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        experiment_name=args.experiment,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
