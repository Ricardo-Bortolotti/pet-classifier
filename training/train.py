"""Training entry point with MLflow experiment tracking."""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import ExperimentConfig, FreezeStrategy
from training.dataset import create_dataloaders
from training.evaluate import evaluate
from training.models import build_model, list_models
from training.optim import build_optimizer, build_scheduler, step_scheduler
from training.transfer_learning import (
    apply_freeze_strategy,
    count_parameters,
    describe_freeze_strategy,
    get_experiment_stage,
    get_training_mode,
)


@dataclass
class TrainingResult:
    """Outcome of a training run."""

    best_val_accuracy: float
    test_accuracy: float
    checkpoint_path: Path | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / max(total, 1)


def _checkpoint_path(config: ExperimentConfig) -> Path:
    suffix = "baseline" if config.model.name == "simple_cnn" else config.model.freeze_strategy.value
    return config.output_dir / f"{config.model.name}_{suffix}_best.pth"


def load_hpo_params(path: Path) -> dict:
    """Load hyperparameters exported by an Optuna study."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def run_training(
    config: ExperimentConfig,
    *,
    save_checkpoint: bool = True,
    log_artifact: bool = True,
    nested_run: bool = False,
    hpo_trial_number: int | None = None,
) -> TrainingResult:
    """Train a model and log the experiment to MLflow."""
    set_seed(config.seed)

    loaders = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    config.model.num_classes = len(loaders.class_names)
    config.model.image_size = config.image_size
    model = build_model(config.model)
    apply_freeze_strategy(model, config.model)

    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model)
    training_mode = get_training_mode(config.model)
    experiment_stage = get_experiment_stage(config.model)
    freeze_description = describe_freeze_strategy(config.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, config.epochs)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    run_kwargs: dict = {"run_name": config.run_name}
    if nested_run:
        run_kwargs["nested"] = True

    with mlflow.start_run(**run_kwargs):
        tags = {
            "training_mode": training_mode,
            "model_name": config.model.name,
            "experiment_stage": experiment_stage,
            "freeze_strategy": config.model.freeze_strategy.value,
        }
        if nested_run:
            tags["hpo"] = "true"
            if hpo_trial_number is not None:
                tags["trial_number"] = str(hpo_trial_number)
                tags["parent_study"] = "hpo-exp3-study"

        mlflow.set_tags(tags)
        mlflow.log_params(
            {
                "model_name": config.model.name,
                "num_classes": config.model.num_classes,
                "pretrained": config.model.pretrained,
                "freeze_strategy": config.model.freeze_strategy.value,
                "freeze_description": freeze_description,
                "training_mode": training_mode,
                "experiment_stage": experiment_stage,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "optimizer": config.optimizer,
                "scheduler": config.scheduler,
                "image_size": config.image_size,
                "val_ratio": config.val_ratio,
                "dropout": config.model.dropout,
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "class_names": ",".join(loaders.class_names),
            }
        )

        print(
            f"\nPetVision AI — Training\n"
            f"Model           : {config.model.name}\n"
            f"Experiment      : {experiment_stage}\n"
            f"Mode            : {training_mode}\n"
            f"Strategy        : {config.model.freeze_strategy.value}\n"
            f"Freezing        : {freeze_description}\n"
            f"Pretrained      : {config.model.pretrained}\n"
            f"Optimizer       : {config.optimizer}\n"
            f"Scheduler       : {config.scheduler}\n"
            f"Parameters      : {trainable_params:,} trainable / {total_params:,} total\n"
            f"Train/Val/Test  : {len(loaders.train.dataset)} / "
            f"{len(loaders.val.dataset)} / {len(loaders.test.dataset)}\n"
        )

        best_accuracy = 0.0
        checkpoint_path: Path | None = None
        if save_checkpoint:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = _checkpoint_path(config)

        for epoch in range(1, config.epochs + 1):
            train_loss = train_one_epoch(model, loaders.train, criterion, optimizer)
            val_metrics = evaluate(model, loaders.val)
            step_scheduler(
                scheduler,
                val_loss=val_metrics["loss"],
                scheduler_name=config.scheduler,
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > best_accuracy and save_checkpoint and checkpoint_path:
                best_accuracy = val_metrics["accuracy"]
                torch.save(
                    {
                        "model_name": config.model.name,
                        "state_dict": model.state_dict(),
                        "class_names": loaders.class_names,
                        "num_classes": config.model.num_classes,
                        "image_size": config.image_size,
                        "pretrained": config.model.pretrained,
                        "freeze_strategy": config.model.freeze_strategy.value,
                        "training_mode": training_mode,
                        "experiment_stage": experiment_stage,
                    },
                    checkpoint_path,
                )
            elif val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]

        test_metrics = evaluate(model, loaders.test)
        mlflow.log_metrics(
            {
                "best_val_accuracy": best_accuracy,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

        print(f"Test | loss={test_metrics['loss']:.4f} | acc={test_metrics['accuracy']:.4f}")

        if log_artifact and checkpoint_path is not None and checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path))

    return TrainingResult(
        best_val_accuracy=best_accuracy,
        test_accuracy=test_metrics["accuracy"],
        checkpoint_path=checkpoint_path if save_checkpoint else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PetVision AI classifier.")
    parser.add_argument(
        "--model",
        choices=list_models(),
        default="simple_cnn",
        help="Model architecture to train.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--experiment", default="petvision-classification")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--freeze-strategy",
        choices=[strategy.value for strategy in FreezeStrategy],
        default=None,
        help="Gradual freezing strategy for EfficientNet-B0.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use ImageNet pretrained weights (default depends on model).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Shortcut for --freeze-strategy head_only/full.",
    )
    parser.add_argument(
        "--from-hpo",
        type=Path,
        default=None,
        help="Load hyperparameters from an Optuna export JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict = {
        "data_dir": args.data_dir,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "experiment_name": args.experiment,
        "run_name": args.run_name,
    }
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    if args.pretrained is not None:
        overrides["pretrained"] = args.pretrained
    if args.freeze_strategy is not None:
        overrides["freeze_strategy"] = args.freeze_strategy
    if args.freeze_backbone is not None:
        overrides["freeze_backbone"] = args.freeze_backbone
    if args.from_hpo is not None:
        overrides.update(load_hpo_params(args.from_hpo))

    config = ExperimentConfig.from_model_name(args.model, **overrides)
    result = run_training(config)
    if result.checkpoint_path is not None:
        print(f"Training complete. Best model saved to: {result.checkpoint_path}")
    else:
        print("Training complete.")


if __name__ == "__main__":
    main()
