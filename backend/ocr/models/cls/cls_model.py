from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ocr_project.stage3_models.angle_classifier import AngleClassifierConfig, MobileNetV3SmallAngleSpec


@dataclass(slots=True)
class ClassifierArtifacts:
    config_path: Path
    weight_path: Path
    output_shapes: dict[str, tuple[int, ...]]


def load_cls_config(config_path: str | Path) -> AngleClassifierConfig:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    model = payload.get("model", {})
    input_cfg = payload.get("input", {})
    inference = payload.get("inference", {})
    normalize = input_cfg.get("normalize", {})
    return AngleClassifierConfig(
        name=model.get("name", "MobileNetV3Small"),
        num_classes=model.get("num_classes", 2),
        pretrained=model.get("pretrained", True),
        pretrained_path=model.get("pretrained_path", "backend/ocr/models/cls/weights/mobilenetv3_small_imagenet.pdparams"),
        dropout=model.get("dropout", 0.2),
        input_height=input_cfg.get("height", 48),
        input_width=input_cfg.get("width", 192),
        normalize_mean=tuple(normalize.get("mean", [0.485, 0.456, 0.406])),
        normalize_std=tuple(normalize.get("std", [0.229, 0.224, 0.225])),
        threshold=inference.get("threshold", 0.5),
        label_list=tuple(inference.get("label_list", ["0", "180"])),
    )


def create_cls_model(config_path: str | Path) -> MobileNetV3SmallAngleSpec:
    return MobileNetV3SmallAngleSpec(load_cls_config(config_path))


def dummy_forward_check(
    config_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 48, 192),
) -> ClassifierArtifacts:
    model = create_cls_model(config_path)
    return ClassifierArtifacts(
        config_path=Path(config_path),
        weight_path=Path(model.config.pretrained_path),
        output_shapes=model.forward_shape(input_shape),
    )
