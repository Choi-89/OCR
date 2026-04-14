from __future__ import annotations


def training_objectives() -> dict[str, str]:
    """OCR-T02: Core loss and optimizer settings."""
    return {
        "detection_loss": "db_loss",
        "recognition_loss": "ctc_loss",
        "optimizer": "adam",
        "scheduler": "cosine_decay",
    }

