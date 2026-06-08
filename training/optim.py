"""Optimizer and learning-rate scheduler factories."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from training.config import ExperimentConfig


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """Build an optimizer for trainable model parameters."""
    params = (param for param in model.parameters() if param.requires_grad)
    name = config.optimizer.lower()

    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )

    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau | None:
    """Build a learning-rate scheduler, or None when scheduler is disabled."""
    name = config.scheduler.lower()

    if name == "none":
        return None
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    raise ValueError(f"Unknown scheduler: {config.scheduler}")


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau | None,
    *,
    val_loss: float,
    scheduler_name: str,
) -> None:
    """Advance the scheduler after each training epoch."""
    if scheduler is None:
        return

    name = scheduler_name.lower()
    if name == "reduce_on_plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()
