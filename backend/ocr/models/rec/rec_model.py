from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ocr_project.stage3_models.recognition_model import RecognitionModelConfig, SVTRTinyModelSpec


@dataclass(slots=True)
class RecognitionArtifacts:
    config_path: Path
    weight_path: Path
    output_shapes: dict[str, tuple[int, ...]]


def load_rec_config(config_path: str | Path) -> RecognitionModelConfig:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    model = payload.get("model", {})
    decoder = payload.get("decoder", {})
    input_cfg = payload.get("input", {})
    single_char = payload.get("single_char", {})
    defaults = RecognitionModelConfig()
    return RecognitionModelConfig(
        name=model.get("name", defaults.name),
        backbone=defaults.backbone.__class__(**model.get("backbone", {})),
        head=defaults.head.__class__(**model.get("head", {})),
        decoder=defaults.decoder.__class__(**decoder),
        input=defaults.input.__class__(**input_cfg),
        single_char=defaults.single_char.__class__(**single_char),
        dict_path=payload.get("dict_path", defaults.dict_path),
    )


def create_rec_model(config_path: str | Path) -> SVTRTinyModelSpec:
    return SVTRTinyModelSpec(load_rec_config(config_path))


def dummy_forward_check(
    config_path: str | Path,
    input_shape: tuple[int, int, int, int] = (4, 3, 32, 128),
) -> RecognitionArtifacts:
    model = create_rec_model(config_path)
    return RecognitionArtifacts(
        config_path=Path(config_path),
        weight_path=Path(model.config.backbone.pretrained_path),
        output_shapes=model.forward_shape(input_shape),
    )


def freeze_strategy(config_path: str | Path) -> list[dict[str, Any]]:
    return create_rec_model(config_path).freeze_plan()
