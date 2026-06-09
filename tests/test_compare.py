from pathlib import Path

import pytest

from training.compare import (
    ModelEvaluationResult,
    checkpoint_label,
    discover_checkpoints,
    select_champion,
)
from training.evaluate import compute_classification_metrics
from training.plots import generate_comparison_plots


def test_compute_classification_metrics_perfect_binary():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 1, 1, 0]
    y_proba = [0.1, 0.9, 0.2, 0.8, 0.95, 0.05]

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_classification_metrics_empty():
    metrics = compute_classification_metrics([], [], [])

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["roc_auc"] == 0.0


def test_checkpoint_label():
    path = Path("app/models/efficientnet_b0_partial_best.pth")
    assert checkpoint_label(path) == "efficientnet_b0_partial"


def test_discover_checkpoints_ignores_subdirectories(tmp_path: Path):
    (tmp_path / "simple_cnn_baseline_best.pth").write_text("x", encoding="utf-8")
    (tmp_path / "efficientnet_b0_partial_best.pth").write_text("x", encoding="utf-8")
    (tmp_path / "hpo").mkdir()
    (tmp_path / "hpo" / "ignored_best.pth").write_text("x", encoding="utf-8")

    checkpoints = discover_checkpoints(tmp_path)

    assert len(checkpoints) == 2
    assert all(checkpoint.parent == tmp_path for checkpoint in checkpoints)


def _make_result(
    label: str,
    val_acc: float,
    test_accuracy: float,
    test_f1: float,
) -> ModelEvaluationResult:
    return ModelEvaluationResult(
        label=label,
        checkpoint_path=Path(f"app/models/{label}_best.pth"),
        experiment_stage=None,
        val_metrics={"accuracy": val_acc},
        test_metrics={
            "accuracy": test_accuracy,
            "precision": test_accuracy,
            "recall": test_accuracy,
            "f1_score": test_f1,
            "roc_auc": test_accuracy,
        },
        test_y_true=[0, 1, 0, 1],
        test_y_pred=[0, 1, 0, 1],
        test_y_proba=[0.1, 0.9, 0.2, 0.8],
    )


def test_select_champion_by_val_acc():
    results = [
        _make_result("model_a", val_acc=0.80, test_accuracy=0.75, test_f1=0.74),
        _make_result("model_b", val_acc=0.92, test_accuracy=0.70, test_f1=0.69),
        _make_result("model_c", val_acc=0.85, test_accuracy=0.88, test_f1=0.87),
    ]

    champion = select_champion(results)

    assert champion.label == "model_b"


def test_select_champion_tiebreakers():
    results = [
        _make_result("model_a", val_acc=0.90, test_accuracy=0.80, test_f1=0.79),
        _make_result("model_b", val_acc=0.90, test_accuracy=0.85, test_f1=0.84),
        _make_result("model_c", val_acc=0.90, test_accuracy=0.85, test_f1=0.90),
    ]

    champion = select_champion(results)

    assert champion.label == "model_c"


def test_select_champion_requires_results():
    with pytest.raises(ValueError, match="No evaluation results"):
        select_champion([])


def test_generate_comparison_plots(tmp_path: Path):
    results = [
        _make_result("model_a", val_acc=0.80, test_accuracy=0.75, test_f1=0.74),
        _make_result("model_b", val_acc=0.92, test_accuracy=0.70, test_f1=0.69),
    ]

    plot_paths = generate_comparison_plots(
        results,
        tmp_path,
        champion_label="model_b",
    )

    assert len(plot_paths) == 3
    for plot_path in plot_paths.values():
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
