from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ocr_project.stage3_models.recognition_model import CRNNConfig, CRNNModelSpec


@dataclass(slots=True)
class CRNNArtifacts:
    config_path: Path
    output_shapes: dict[str, tuple[int, ...]]


def create_crnn_model(config: CRNNConfig | None = None) -> CRNNModelSpec:
    return CRNNModelSpec(config or CRNNConfig())


def dummy_forward_check(
    input_shape: tuple[int, int, int, int] = (4, 3, 32, 128),
) -> CRNNArtifacts:
    model = create_crnn_model()
    return CRNNArtifacts(
        config_path=Path("backend/ocr/models/rec/rec_config.yaml"),
        output_shapes=model.forward_shape(input_shape),
    )
