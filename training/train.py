"""Training entry point with MLflow experiment tracking."""

import argparse
import random
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import ExperimentConfig
from training.dataset import create_dataloaders
from training.evaluate import evaluate
from training.models import build_model, list_models


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / max(total, 1)


def run_training(config: ExperimentConfig) -> Path:
    """Train a model and log the experiment to MLflow."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = build_model(config.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(
            {
                "model_name": config.model.name,
                "num_classes": config.model.num_classes,
                "pretrained": config.model.pretrained,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "image_size": config.image_size,
                "val_ratio": config.val_ratio,
                "class_names": ",".join(loaders.class_names),
            }
        )

        best_accuracy = 0.0
        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.output_dir / "best_model.pth"

        for epoch in range(1, config.epochs + 1):
            train_loss = train_one_epoch(model, loaders.train, criterion, optimizer, device)
            val_metrics = evaluate(model, loaders.val, device)

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

            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                torch.save(
                    {
                        "model_name": config.model.name,
                        "state_dict": model.state_dict(),
                        "class_names": loaders.class_names,
                        "num_classes": config.model.num_classes,
                        "image_size": config.image_size,
                    },
                    checkpoint_path,
                )

        test_metrics = evaluate(model, loaders.test, device)
        mlflow.log_metrics(
            {
                "best_val_accuracy": best_accuracy,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

        print(f"Test | loss={test_metrics['loss']:.4f} | acc={test_metrics['accuracy']:.4f}")

        mlflow.log_artifact(str(checkpoint_path))

    return checkpoint_path


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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--experiment", default="petvision-classification")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_model_name(
        args.model,
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        experiment_name=args.experiment,
        run_name=args.run_name,
    )
    checkpoint = run_training(config)
    print(f"Training complete. Best model saved to: {checkpoint}")


if __name__ == "__main__":
    main()
