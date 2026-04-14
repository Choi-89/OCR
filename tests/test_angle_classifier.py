from __future__ import annotations

from pathlib import Path

from backend.ocr.models.cls.cls_model import dummy_forward_check
from ocr_project.stage3_models.angle_classifier import (
    AngleClassifierConfig,
    MobileNetV3SmallAngleSpec,
    build_angle_dataset,
    classify_orientation,
)


CONFIG_PATH = Path("C:/OCR/backend/ocr/models/cls/cls_config.yaml")


def test_angle_forward_shape_matches_binary_classifier() -> None:
    spec = MobileNetV3SmallAngleSpec(AngleClassifierConfig())
    outputs = spec.forward_shape((1, 3, 48, 192))
    assert outputs["logits"] == (1, 2)
    assert outputs["probabilities"] == (1, 2)


def test_dummy_forward_check_returns_expected_weight_path() -> None:
    artifacts = dummy_forward_check(CONFIG_PATH)
    assert artifacts.output_shapes["logits"] == (1, 2)
    assert "mobilenetv3_small_imagenet.pdparams" in str(artifacts.weight_path)


def test_classify_orientation_marks_rotation_for_180() -> None:
    result = classify_orientation((0.1, 0.9), threshold=0.5)
    assert result["label"] == "180"
    assert result["rotate"] is True


def test_build_angle_dataset_creates_split_structure(tmp_path: Path) -> None:
    source = tmp_path / "masked"
    source.mkdir()
    sample = source / "sample.png"
    sample.write_bytes(b"fake")
    counts = build_angle_dataset(source, tmp_path / "angle")
    assert counts["train_0"] + counts["val_0"] + counts["test_0"] == 1
