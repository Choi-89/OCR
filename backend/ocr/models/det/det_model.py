from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ocr_project.stage3_models.detection_model import DBNetPPModelSpec, DetectionModelConfig


@dataclass(slots=True)
class DetectionModelArtifacts:
    config_path: Path
    weight_path: Path
    output_shapes: dict[str, tuple[int, int, int, int]]


def load_det_config(config_path: str | Path) -> DetectionModelConfig:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    model = payload.get("model", {})
    inference = payload.get("inference", {})
    input_cfg = payload.get("input", {})
    return DetectionModelConfig(
        name=model.get("name", "DBNet++"),
        backbone=DetectionModelConfig().backbone.__class__(**model.get("backbone", {})),
        neck=DetectionModelConfig().neck.__class__(**model.get("neck", {})),
        head=DetectionModelConfig().head.__class__(**model.get("head", {})),
        inference=DetectionModelConfig().inference.__class__(**inference),
        input=DetectionModelConfig().input.__class__(**input_cfg),
    )


def create_det_model(config_path: str | Path) -> DBNetPPModelSpec:
    return DBNetPPModelSpec(load_det_config(config_path))


def dummy_forward_check(
    config_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 960, 960),
) -> DetectionModelArtifacts:
    model = create_det_model(config_path)
    return DetectionModelArtifacts(
        config_path=Path(config_path),
        weight_path=Path(model.config.backbone.pretrained_path),
        output_shapes=model.forward_shape(input_shape),
    )


def freeze_strategy(config_path: str | Path) -> list[dict[str, Any]]:
    return create_det_model(config_path).freeze_plan()
