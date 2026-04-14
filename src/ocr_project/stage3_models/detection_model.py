from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BackboneConfig:
    type: str = "ResNet50"
    pretrained: bool = True
    pretrained_path: str = "backend/ocr/models/det/weights/resnet50_imagenet.pdparams"
    freeze_stages: int = 4
    stage_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048)
    stage_strides: tuple[int, int, int, int] = (4, 8, 16, 32)
    learning_rate_scale: float = 0.1


@dataclass(slots=True)
class NeckConfig:
    type: str = "FPNASF"
    in_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048)
    out_channels: int = 256


@dataclass(slots=True)
class HeadConfig:
    type: str = "DBHead"
    k: int = 50
    use_bias: bool = False


@dataclass(slots=True)
class InferenceConfig:
    prob_threshold: float = 0.3
    box_threshold: float = 0.5
    min_box_size: int = 3
    unclip_ratio: float = 1.5
    max_candidates: int = 1000


@dataclass(slots=True)
class InputConfig:
    max_side: int = 960
    divisor: int = 32
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class DetectionModelConfig:
    name: str = "DBNet++"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    input: InputConfig = field(default_factory=InputConfig)


@dataclass(slots=True)
class FeatureMapSpec:
    name: str
    channels: int
    stride: int
    shape: tuple[int, int, int, int]


class DBNetPPModelSpec:
    """Architecture spec and shape simulator for ShiftFlow text detection."""

    def __init__(self, config: DetectionModelConfig | None = None):
        self.config = config or DetectionModelConfig()
        self.backbone_frozen = self.config.backbone.freeze_stages > 0

    def freeze_backbone(self) -> None:
        self.backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        self.backbone_frozen = False

    def freeze_plan(self) -> list[dict[str, Any]]:
        return [
            {
                "phase": 1,
                "epochs": 5,
                "backbone_frozen": True,
                "learning_rate_scale": self.config.backbone.learning_rate_scale,
                "trainable_modules": ["neck", "head"],
            },
            {
                "phase": 2,
                "epochs": "remaining",
                "backbone_frozen": False,
                "learning_rate_scale": self.config.backbone.learning_rate_scale,
                "trainable_modules": ["backbone", "neck", "head"],
            },
        ]

    def validate_input_shape(self, shape: tuple[int, int, int, int]) -> list[str]:
        issues: list[str] = []
        _, channels, height, width = shape
        if channels != 3:
            issues.append(f"expected_3_channels_got_{channels}")
        if max(height, width) > self.config.input.max_side:
            issues.append(f"max_side_exceeds_{self.config.input.max_side}")
        if height % self.config.input.divisor != 0 or width % self.config.input.divisor != 0:
            issues.append(f"input_not_divisible_by_{self.config.input.divisor}")
        return issues

    def backbone_feature_shapes(self, input_shape: tuple[int, int, int, int]) -> list[FeatureMapSpec]:
        batch, _, height, width = input_shape
        specs: list[FeatureMapSpec] = []
        for index, (channels, stride) in enumerate(
            zip(self.config.backbone.stage_channels, self.config.backbone.stage_strides),
            start=2,
        ):
            specs.append(
                FeatureMapSpec(
                    name=f"C{index}",
                    channels=channels,
                    stride=stride,
                    shape=(batch, channels, height // stride, width // stride),
                )
            )
        return specs

    def neck_feature_shapes(self, input_shape: tuple[int, int, int, int]) -> list[FeatureMapSpec]:
        batch, _, height, width = input_shape
        return [
            FeatureMapSpec(
                name=f"P{index}",
                channels=self.config.neck.out_channels,
                stride=stride,
                shape=(batch, self.config.neck.out_channels, height // stride, width // stride),
            )
            for index, stride in enumerate(self.config.backbone.stage_strides, start=2)
        ]

    def fused_feature_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        batch, _, height, width = input_shape
        return (batch, self.config.neck.out_channels, height // 4, width // 4)

    def forward_shape(self, input_shape: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
        fused = self.fused_feature_shape(input_shape)
        return {
            "probability_map": (fused[0], 1, fused[2], fused[3]),
            "threshold_map": (fused[0], 1, fused[2], fused[3]),
            "binary_map": (fused[0], 1, fused[2], fused[3]),
        }

    def model_summary(self, input_shape: tuple[int, int, int, int] = (1, 3, 960, 960)) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "backbone": self.config.backbone.type,
            "neck": self.config.neck.type,
            "head": self.config.head.type,
            "input_shape": input_shape,
            "input_issues": self.validate_input_shape(input_shape),
            "backbone_features": [feature.__dict__ for feature in self.backbone_feature_shapes(input_shape)],
            "neck_features": [feature.__dict__ for feature in self.neck_feature_shapes(input_shape)],
            "fused_feature": self.fused_feature_shape(input_shape),
            "outputs": self.forward_shape(input_shape),
            "freeze_plan": self.freeze_plan(),
        }


def build_detection_model(config: DetectionModelConfig | None = None) -> dict[str, Any]:
    """OCR-M01: Build a DBNet++ architecture summary for text detection."""
    spec = DBNetPPModelSpec(config or DetectionModelConfig())
    return spec.model_summary()


def expected_weight_path(config: DetectionModelConfig | None = None) -> Path:
    cfg = config or DetectionModelConfig()
    return Path(cfg.backbone.pretrained_path)
