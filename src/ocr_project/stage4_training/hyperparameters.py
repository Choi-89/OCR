from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HyperParameters:
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.001
    warmup_steps: int = 500
    weight_decay: float = 0.0001


def default_hyperparameters() -> HyperParameters:
    """OCR-T03: Default training hyperparameters."""
    return HyperParameters()

