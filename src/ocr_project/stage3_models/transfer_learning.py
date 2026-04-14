from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TransferLearningPlan:
    pretrained_checkpoint: str
    freeze_layers: list[str]
    finetune_layers: list[str]


def default_transfer_plan() -> TransferLearningPlan:
    """OCR-M04: Freeze lower layers and fine-tune task-specific heads."""
    return TransferLearningPlan(
        pretrained_checkpoint="paddleocr_base",
        freeze_layers=["stem", "stage1"],
        finetune_layers=["stage2", "stage3", "head"],
    )

