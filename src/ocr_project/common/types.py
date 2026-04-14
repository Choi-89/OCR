from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetItem:
    image_path: Path
    split: str = "train"
    text: str = ""
    boxes: list[list[int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionResult:
    boxes: list[list[int]]
    scores: list[float]


@dataclass(slots=True)
class RecognitionResult:
    text: str
    confidence: float


@dataclass(slots=True)
class ExperimentConfig:
    project_name: str = "ocr-workflow"
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "artifacts"

