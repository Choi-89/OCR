from __future__ import annotations


def monitoring_targets() -> list[str]:
    """OCR-T04: Signals to monitor during training."""
    return ["loss_curve", "checkpoint", "wandb", "validation_accuracy"]

