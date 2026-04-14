from __future__ import annotations


def confidence_ui_fields() -> list[str]:
    """OCR-S03: UI bindings for OCR confidence and feedback."""
    return ["confidence_score", "field_level_confidence", "user_feedback"]

