"""Hyperparameter optimization with Optuna for experiment 3 (partial fine-tuning)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import optuna

from training.config import ExperimentConfig, FreezeStrategy
from training.train import run_training

DEFAULT_STUDY_NAME = "exp3_partial_hpo"
DEFAULT_STORAGE = "sqlite:///optuna_exp3.db"
DEFAULT_EXPORT_PATH = Path("app/models/hpo/exp3_best_params.json")
PARENT_RUN_NAME = "hpo-exp3-study"


def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    """Sample hyperparameters for a single Optuna trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
        "scheduler": trial.suggest_categorical(
            "scheduler",
            ["none", "step", "cosine", "reduce_on_plateau"],
        ),
    }


def export_best_params(study: optuna.Study, export_path: Path) -> Path:
    """Write the best trial hyperparameters to JSON."""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)
        file.write("\n")
    return export_path


def run_hpo_study(
    *,
    n_trials: int = 10,
    epochs: int = 5,
    data_dir: Path = Path("data"),
    experiment_name: str = "petvision-classification",
    study_name: str = DEFAULT_STUDY_NAME,
    storage: str = DEFAULT_STORAGE,
    export_path: Path = DEFAULT_EXPORT_PATH,
    seed: int = 42,
) -> optuna.Study:
    """Run Optuna HPO for EfficientNet-B0 partial fine-tuning (experiment 3)."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        trial_number = trial.number + 1
        params = suggest_hyperparameters(trial)
        config = ExperimentConfig.from_model_name(
            "efficientnet_b0",
            freeze_strategy=FreezeStrategy.PARTIAL,
            data_dir=data_dir,
            epochs=epochs,
            experiment_name=experiment_name,
            run_name=f"trial_{trial_number:03d}",
            seed=seed,
            **params,
        )

        result = run_training(
            config,
            save_checkpoint=False,
            log_artifact=False,
            nested_run=True,
            hpo_trial_number=trial_number,
        )
        return result.best_val_accuracy

    with mlflow.start_run(run_name=PARENT_RUN_NAME):
        mlflow.set_tags(
            {
                "hpo": "true",
                "experiment_stage": "exp3_partial_finetuning",
                "parent_study": PARENT_RUN_NAME,
            }
        )
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "epochs_per_trial": epochs,
                "freeze_strategy": FreezeStrategy.PARTIAL.value,
                "model_name": "efficientnet_b0",
                "study_name": study_name,
            }
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
        )
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params(
            {
                "best_trial": study.best_trial.number + 1,
                **study.best_params,
            }
        )
        mlflow.log_metrics(
            {
                "best_val_accuracy": study.best_value,
            }
        )

        exported = export_best_params(study, export_path)
        mlflow.log_artifact(str(exported))

        print(
            f"\nHPO complete — best trial: trial_{study.best_trial.number + 1:03d}\n"
            f"best_val_accuracy={study.best_value:.4f}\n"
            f"Exported parameters: {exported}\n"
            f"\nSuggested final training:\n"
            f"  uv run python -m training.train --model efficientnet_b0 "
            f"--freeze-strategy partial --epochs 10 "
            f"--run-name exp3-optimized --from-hpo {exported}\n"
        )

    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for experiment 3 (partial fine-tuning).",
    )
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--experiment", default="petvision-classification")
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--export-path", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_hpo_study(
        n_trials=args.n_trials,
        epochs=args.epochs,
        data_dir=args.data_dir,
        experiment_name=args.experiment,
        study_name=args.study_name,
        storage=args.storage,
        export_path=args.export_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
