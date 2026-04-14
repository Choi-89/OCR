from __future__ import annotations

from pathlib import Path


def export_inference_model(checkpoint_dir: Path, output_dir: Path) -> Path:
    """OCR-S01: Export an inference-ready model artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

