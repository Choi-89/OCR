from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingHealthState:
    recent_losses: list[float] = field(default_factory=list)
    loss_window: int = 10


def check_training_health(
    loss: float,
    grad_norm: float | None,
    step: int,
    state: TrainingHealthState | None = None,
) -> list[str]:
    state = state or TrainingHealthState()
    messages: list[str] = []
    if math.isnan(float(loss)):
        raise RuntimeError(f"NaN loss at step {step}. Stop training.")

    state.recent_losses.append(float(loss))
    if len(state.recent_losses) > state.loss_window:
        state.recent_losses.pop(0)

    if len(state.recent_losses) == state.loss_window and all(
        right > left for left, right in zip(state.recent_losses, state.recent_losses[1:])
    ):
        raise RuntimeError(f"Loss diverged for {state.loss_window} logged steps at step {step}.")

    if grad_norm is not None and grad_norm > 1000:
        messages.append(f"grad_norm_spike:{grad_norm:.3f}:step:{step}")
    return messages


def check_epoch_health(epoch: int, train_loss: float, val_loss: float) -> list[str]:
    if epoch >= 5 and val_loss > train_loss * 10:
        return [f"val_loss_gap:epoch:{epoch}:train:{train_loss:.4f}:val:{val_loss:.4f}"]
    return []
