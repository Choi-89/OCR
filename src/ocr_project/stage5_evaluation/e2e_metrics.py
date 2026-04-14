from __future__ import annotations


def end_to_end_metric_names() -> list[str]:
    """OCR-E03: End-to-end payroll extraction metrics."""
    return ["full_match_rate", "date_match_rate", "name_match_rate", "code_match_rate"]

