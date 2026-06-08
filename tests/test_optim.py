import json
from pathlib import Path

import torch
import torch.nn as nn

from training.config import ExperimentConfig, ModelConfig
from training.models import build_model
from training.optim import build_optimizer, build_scheduler, step_scheduler
from training.train import load_hpo_params
from training.tune import export_best_params, suggest_hyperparameters


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _experiment_config(**overrides) -> ExperimentConfig:
    config = ExperimentConfig(
        model=ModelConfig(name="simple_cnn", num_classes=2, pretrained=False),
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
    return config


def test_build_optimizer_types():
    model = _DummyModel()
    for name, expected_type in [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("sgd", torch.optim.SGD),
    ]:
        config = _experiment_config(optimizer=name)
        optimizer = build_optimizer(model, config)
        assert isinstance(optimizer, expected_type)


def test_build_scheduler_none():
    model = _DummyModel()
    config = _experiment_config(scheduler="none")
    optimizer = build_optimizer(model, config)
    assert build_scheduler(optimizer, config, epochs=5) is None


def test_build_scheduler_types():
    model = _DummyModel()
    optimizer = build_optimizer(model, _experiment_config())
    for scheduler_name, expected_type in [
        ("step", torch.optim.lr_scheduler.StepLR),
        ("cosine", torch.optim.lr_scheduler.CosineAnnealingLR),
        ("reduce_on_plateau", torch.optim.lr_scheduler.ReduceLROnPlateau),
    ]:
        config = _experiment_config(scheduler=scheduler_name)
        scheduler = build_scheduler(optimizer, config, epochs=5)
        assert isinstance(scheduler, expected_type)


def test_step_scheduler_does_not_raise():
    model = build_model(ModelConfig(name="simple_cnn", num_classes=2, pretrained=False))
    config = _experiment_config(scheduler="cosine")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, epochs=3)

    for _ in range(1, 4):
        step_scheduler(scheduler, val_loss=0.5, scheduler_name=config.scheduler)

    config.scheduler = "reduce_on_plateau"
    plateau_scheduler = build_scheduler(optimizer, config, epochs=3)
    step_scheduler(plateau_scheduler, val_loss=0.5, scheduler_name=config.scheduler)
    step_scheduler(None, val_loss=0.5, scheduler_name="none")


def test_load_hpo_params_roundtrip(tmp_path: Path):
    params = {
        "learning_rate": 0.0003,
        "batch_size": 32,
        "dropout": 0.25,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": "cosine",
    }
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps(params), encoding="utf-8")

    loaded = load_hpo_params(path)
    assert loaded == params


def test_export_best_params(tmp_path: Path):
    study = optuna_create_stub_study()
    export_path = tmp_path / "hpo" / "exp3_best_params.json"

    exported = export_best_params(study, export_path)

    assert exported.exists()
    assert json.loads(exported.read_text(encoding="utf-8")) == study.best_params


def test_suggest_hyperparameters_keys():
    trial = _FakeTrial()
    params = suggest_hyperparameters(trial)

    assert set(params) == {
        "learning_rate",
        "batch_size",
        "dropout",
        "weight_decay",
        "optimizer",
        "scheduler",
    }


class _FakeTrial:
    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        return 1e-3

    def suggest_categorical(self, name: str, choices: list) -> object:
        return choices[0]


def optuna_create_stub_study():
    import optuna

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(
        {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "dropout": 0.2,
            "weight_decay": 1e-4,
            "optimizer": "adam",
            "scheduler": "none",
        }
    )
    study.optimize(lambda trial: 0.9, n_trials=1)
    return study
