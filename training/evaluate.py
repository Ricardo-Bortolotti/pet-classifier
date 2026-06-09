import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
) -> tuple[list[int], list[int], list[float]]:
    """Collect ground truth, predictions and positive-class probabilities."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[float] = []

    for images, labels in dataloader:
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        if probabilities.shape[1] == 2:
            positive_probs = probabilities[:, 1]
        else:
            positive_probs = probabilities.max(dim=1).values

        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).tolist())
        y_proba.extend(positive_probs.tolist())

    return y_true, y_pred, y_proba


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_proba: list[float] | list[list[float]],
) -> dict[str, float]:
    """Compute classification metrics for model comparison."""
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
        }

    num_classes = len(set(y_true))
    if num_classes > 2 and y_proba and isinstance(y_proba[0], list):
        roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    else:
        flat_proba = [float(value) for value in y_proba]
        roc_auc = float(roc_auc_score(y_true, flat_proba))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
) -> dict[str, float]:
    """Compute loss and accuracy on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    dataloader: DataLoader,
) -> dict[str, float]:
    """Compute loss and full classification metrics on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[float] = []

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        probabilities = F.softmax(outputs, dim=1)
        if probabilities.shape[1] == 2:
            positive_probs = probabilities[:, 1]
        else:
            positive_probs = probabilities.max(dim=1).values

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).tolist())
        y_proba.extend(positive_probs.tolist())

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
        }

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    metrics["loss"] = total_loss / total
    return metrics
