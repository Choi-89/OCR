from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AngleClassifierConfig:
    name: str = "MobileNetV3Small"
    num_classes: int = 2
    pretrained: bool = True
    pretrained_path: str = "backend/ocr/models/cls/weights/mobilenetv3_small_imagenet.pdparams"
    dropout: float = 0.2
    input_height: int = 48
    input_width: int = 192
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    threshold: float = 0.5
    label_list: tuple[str, str] = ("0", "180")


@dataclass(slots=True)
class AngleDatasetPlan:
    source_dir: str = "data/masked"
    output_dir: str = "data/angle"
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    class_labels: tuple[str, str] = ("0", "1")


class MobileNetV3SmallAngleSpec:
    """Architecture and shape simulator for 0/180 degree classification."""

    def __init__(self, config: AngleClassifierConfig | None = None):
        self.config = config or AngleClassifierConfig()

    def validate_input_shape(self, shape: tuple[int, int, int, int]) -> list[str]:
        issues: list[str] = []
        _, channels, height, width = shape
        if channels != 3:
            issues.append(f"expected_3_channels_got_{channels}")
        if height != self.config.input_height:
            issues.append(f"expected_height_{self.config.input_height}_got_{height}")
        if width != self.config.input_width:
            issues.append(f"expected_width_{self.config.input_width}_got_{width}")
        return issues

    def feature_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        batch, _, height, width = input_shape
        return (batch, 576, max(1, height // 32), max(1, width // 32))

    def pooled_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int]:
        batch, channels, _, _ = self.feature_shape(input_shape)
        return (batch, channels)

    def hidden_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int]:
        batch, _ = self.pooled_shape(input_shape)
        return (batch, 128)

    def forward_shape(self, input_shape: tuple[int, int, int, int]) -> dict[str, tuple[int, ...]]:
        batch = input_shape[0]
        return {
            "logits": (batch, self.config.num_classes),
            "probabilities": (batch, self.config.num_classes),
        }

    def model_summary(self, input_shape: tuple[int, int, int, int] = (1, 3, 48, 192)) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "input_shape": input_shape,
            "input_issues": self.validate_input_shape(input_shape),
            "feature_shape": self.feature_shape(input_shape),
            "pooled_shape": self.pooled_shape(input_shape),
            "hidden_shape": self.hidden_shape(input_shape),
            "outputs": self.forward_shape(input_shape),
            "threshold": self.config.threshold,
            "labels": self.config.label_list,
        }


def build_angle_classifier(config: AngleClassifierConfig | None = None) -> dict[str, Any]:
    spec = MobileNetV3SmallAngleSpec(config or AngleClassifierConfig())
    return spec.model_summary()


def classify_orientation(probabilities: tuple[float, float], threshold: float = 0.5) -> dict[str, Any]:
    prob_0, prob_180 = probabilities
    predicted = "180" if prob_0 < threshold and prob_180 >= prob_0 else "0"
    return {
        "label": predicted,
        "rotate": predicted == "180",
        "probabilities": {"0": prob_0, "180": prob_180},
    }


def build_angle_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, int]:
    """Create 0/180 dataset folders by writing original and 180-degree variants."""
    source = Path(source_dir)
    output = Path(output_dir)
    images = sorted(path for path in source.glob("*.*") if path.is_file())
    counts = {"train_0": 0, "train_1": 0, "val_0": 0, "val_1": 0, "test_0": 0, "test_1": 0}

    train_end = int(len(images) * train_ratio)
    val_end = train_end + int(len(images) * val_ratio)
    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, split_images in split_map.items():
        for label in ("0", "1"):
            (output / split_name / label).mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            normal_target = output / split_name / "0" / image_path.name
            rotated_target = output / split_name / "1" / f"{image_path.stem}_rot180{image_path.suffix}"
            shutil.copy2(image_path, normal_target)
            _write_rotated_180(image_path, rotated_target)
            counts[f"{split_name}_0"] += 1
            counts[f"{split_name}_1"] += 1
    return counts


def _write_rotated_180(source: Path, target: Path) -> None:
    """Persist a 180-degree variant, falling back to a copy when image libs are unavailable."""
    try:
        from PIL import Image

        with Image.open(source) as image:
            image.rotate(180, expand=True).save(target)
        return
    except Exception:
        shutil.copy2(source, target)
