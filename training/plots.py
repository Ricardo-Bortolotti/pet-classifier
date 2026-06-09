"""Comparative plots for final model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


class ComparableModelResult(Protocol):
    label: str
    val_acc: float
    test_metrics: dict[str, float]
    test_y_true: list[int]
    test_y_proba: list[float]


METRIC_LABELS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1"),
    ("roc_auc", "ROC-AUC"),
]


def plot_metrics_comparison(
    results: list[ComparableModelResult],
    output_path: Path,
    *,
    champion_label: str | None = None,
) -> Path:
    """Grouped bar chart of test metrics across models."""
    labels = [result.label for result in results]
    metric_names = [label for _, label in METRIC_LABELS]
    x_positions = range(len(metric_names))
    bar_width = 0.8 / max(len(labels), 1)

    fig, axis = plt.subplots(figsize=(12, 6))
    for index, result in enumerate(results):
        offsets = [pos + (index - (len(labels) - 1) / 2) * bar_width for pos in x_positions]
        values = [result.test_metrics[key] for key, _ in METRIC_LABELS]
        color = "#2ca02c" if result.label == champion_label else None
        axis.bar(offsets, values, width=bar_width, label=result.label, color=color)

    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(metric_names)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Comparativo de métricas no conjunto de teste")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_val_acc_comparison(
    results: list[ComparableModelResult],
    output_path: Path,
    *,
    champion_label: str | None = None,
) -> Path:
    """Horizontal bar chart ordered by validation accuracy."""
    sorted_results = sorted(results, key=lambda item: item.val_acc, reverse=True)
    labels = [result.label for result in sorted_results]
    values = [result.val_acc for result in sorted_results]
    colors = ["#2ca02c" if label == champion_label else "#1f77b4" for label in labels]

    fig, axis = plt.subplots(figsize=(10, max(4, len(labels) * 0.6)))
    axis.barh(labels, values, color=colors)
    axis.set_xlim(0.0, 1.05)
    axis.set_xlabel("val_acc")
    axis.set_title("Comparativo de val_acc (critério do Champion Model)")
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_roc_curves(
    results: list[ComparableModelResult],
    output_path: Path,
    *,
    champion_label: str | None = None,
) -> Path:
    """ROC curves for all models on the test set."""
    fig, axis = plt.subplots(figsize=(8, 6))

    for result in results:
        if not result.test_y_true:
            continue
        fpr, tpr, _ = roc_curve(result.test_y_true, result.test_y_proba)
        roc_auc = auc(fpr, tpr)
        linewidth = 2.5 if result.label == champion_label else 1.5
        axis.plot(
            fpr,
            tpr,
            linewidth=linewidth,
            label=f"{result.label} (AUC={roc_auc:.3f})",
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Curvas ROC no conjunto de teste")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_comparison_plots(
    results: list[ComparableModelResult],
    output_dir: Path,
    *,
    champion_label: str | None = None,
) -> dict[str, Path]:
    """Generate all comparison plots and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "metrics_comparison_test": plot_metrics_comparison(
            results,
            output_dir / "metrics_comparison_test.png",
            champion_label=champion_label,
        ),
        "val_acc_comparison": plot_val_acc_comparison(
            results,
            output_dir / "val_acc_comparison.png",
            champion_label=champion_label,
        ),
        "roc_curves_test": plot_roc_curves(
            results,
            output_dir / "roc_curves_test.png",
            champion_label=champion_label,
        ),
    }
