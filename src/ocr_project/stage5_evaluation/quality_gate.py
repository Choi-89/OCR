from __future__ import annotations


def passes_quality_gate(cer: float, threshold: float = 0.03) -> bool:
    """OCR-E04: Gate for pass/retrain decisions."""
    return cer < threshold

