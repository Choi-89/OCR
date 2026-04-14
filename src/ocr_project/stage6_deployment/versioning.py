from __future__ import annotations


def release_controls() -> list[str]:
    """OCR-S04: Versioning, rollout, and rollback policies."""
    return ["model_registry", "ab_test", "rollback_policy"]

