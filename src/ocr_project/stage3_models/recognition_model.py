from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocr_project.stage2_preprocess.korean_charset import estimate_vocab_size


@dataclass(slots=True)
class SVTRBackboneConfig:
    type: str = "SVTRNet"
    img_size: tuple[int, int] = (32, 320)
    in_channels: int = 3
    embed_dim: tuple[int, int, int] = (64, 128, 256)
    depth: tuple[int, int, int] = (3, 6, 3)
    num_heads: tuple[int, int, int] = (2, 4, 8)
    mixer: tuple[str, ...] = (
        "Local",
        "Local",
        "Local",
        "Global",
        "Global",
        "Global",
        "Global",
        "Global",
        "Global",
        "Local",
        "Local",
        "Local",
    )
    local_mixer: tuple[tuple[int, int], ...] = ((7, 11), (7, 11), (7, 11))
    pretrained: bool = True
    pretrained_path: str = "backend/ocr/models/rec/weights/svtr_tiny_pretrained.pdparams"
    patch_embedding_freeze_epochs: int = 5
    patch_embedding_lr_scale: float = 0.1


@dataclass(slots=True)
class CTCHeadConfig:
    type: str = "CTCHead"
    vocab_size: int = field(default_factory=estimate_vocab_size)
    mid_channels: int = 96


@dataclass(slots=True)
class DecoderConfig:
    type: str = "CTCDecoder"
    method: str = "greedy"
    beam_width: int = 5
    blank_index: int = field(default_factory=lambda: estimate_vocab_size() - 1)


@dataclass(slots=True)
class RecognitionInputConfig:
    height: int = 32
    max_width: int = 320
    min_width: int = 32
    padding_value: int = 0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class SingleCharConfig:
    enabled: bool = True
    max_output_chars: int = 1


@dataclass(slots=True)
class RecognitionModelConfig:
    name: str = "SVTR_Tiny"
    backbone: SVTRBackboneConfig = field(default_factory=SVTRBackboneConfig)
    head: CTCHeadConfig = field(default_factory=CTCHeadConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    input: RecognitionInputConfig = field(default_factory=RecognitionInputConfig)
    single_char: SingleCharConfig = field(default_factory=SingleCharConfig)
    dict_path: str = "backend/ocr/dict/dict_latest.txt"


@dataclass(slots=True)
class CRNNConfig:
    cnn_backbone: str = "VGG"
    cnn_pretrained: bool = True
    cnn_pretrained_path: str = "backend/ocr/models/rec/weights/resnet34_imagenet.pdparams"
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    vocab_size: int = field(default_factory=estimate_vocab_size)
    blank_index: int = field(default_factory=lambda: estimate_vocab_size() - 1)


class SVTRTinyModelSpec:
    """Architecture and shape simulator for SVTR-Tiny recognition."""

    def __init__(self, config: RecognitionModelConfig | None = None):
        self.config = config or RecognitionModelConfig()

    def validate_input_shape(self, shape: tuple[int, int, int, int]) -> list[str]:
        issues: list[str] = []
        _, channels, height, width = shape
        if channels != self.config.backbone.in_channels:
            issues.append(f"expected_channels_{self.config.backbone.in_channels}_got_{channels}")
        if height != self.config.input.height:
            issues.append(f"expected_height_{self.config.input.height}_got_{height}")
        if width < self.config.input.min_width:
            issues.append(f"width_below_min_{self.config.input.min_width}")
        if width > self.config.input.max_width:
            issues.append(f"width_above_max_{self.config.input.max_width}")
        return issues

    def patch_embedding_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        batch, _, height, width = input_shape
        return (batch, 128, height // 4, max(1, width // 4))

    def sequence_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int, int]:
        batch, _, _, width = input_shape
        return (batch, max(1, (width // 4) * 8), 128)

    def combining_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int, int]:
        batch, _, _, width = input_shape
        return (batch, max(1, width // 4), self.config.head.mid_channels)

    def forward_shape(self, input_shape: tuple[int, int, int, int]) -> dict[str, tuple[int, ...]]:
        batch, _, _, width = input_shape
        return {"logits": (batch, max(1, width // 4), self.config.head.vocab_size)}

    def freeze_plan(self) -> list[dict[str, Any]]:
        return [
            {
                "phase": 1,
                "epochs": self.config.backbone.patch_embedding_freeze_epochs,
                "patch_embedding_frozen": True,
                "learning_rate_scale": self.config.backbone.patch_embedding_lr_scale,
                "trainable_modules": ["mixing_blocks", "ctc_head"],
            },
            {
                "phase": 2,
                "epochs": "remaining",
                "patch_embedding_frozen": False,
                "learning_rate_scale": self.config.backbone.patch_embedding_lr_scale,
                "trainable_modules": ["patch_embedding", "mixing_blocks", "ctc_head"],
            },
        ]

    def model_summary(self, input_shape: tuple[int, int, int, int] = (4, 3, 32, 128)) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "input_shape": input_shape,
            "input_issues": self.validate_input_shape(input_shape),
            "patch_embedding": self.patch_embedding_shape(input_shape),
            "sequence": self.sequence_shape(input_shape),
            "combining": self.combining_shape(input_shape),
            "outputs": self.forward_shape(input_shape),
            "vocab_size": self.config.head.vocab_size,
            "blank_index": self.config.decoder.blank_index,
            "freeze_plan": self.freeze_plan(),
        }


class CRNNModelSpec:
    """Architecture and shape simulator for CRNN recognition."""

    def __init__(self, config: CRNNConfig | None = None):
        self.config = config or CRNNConfig()

    def cnn_feature_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        batch, _, _, width = input_shape
        return (batch, 512, 1, max(1, width // 4))

    def sequence_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int, int]:
        batch, _, _, width = input_shape
        return (batch, max(1, width // 4), 512)

    def forward_shape(self, input_shape: tuple[int, int, int, int]) -> dict[str, tuple[int, ...]]:
        batch, _, _, width = input_shape
        return {"logits": (batch, max(1, width // 4), self.config.vocab_size)}

    def model_summary(self, input_shape: tuple[int, int, int, int] = (4, 3, 32, 128)) -> dict[str, Any]:
        return {
            "name": "CRNN",
            "cnn_feature": self.cnn_feature_shape(input_shape),
            "sequence": self.sequence_shape(input_shape),
            "outputs": self.forward_shape(input_shape),
            "vocab_size": self.config.vocab_size,
            "blank_index": self.config.blank_index,
            "rnn_hidden_size": self.config.rnn_hidden_size,
            "rnn_layers": self.config.rnn_layers,
        }


def build_recognition_model(config: RecognitionModelConfig | None = None) -> dict[str, Any]:
    return SVTRTinyModelSpec(config or RecognitionModelConfig()).model_summary()


def build_crnn_model(config: CRNNConfig | None = None) -> dict[str, Any]:
    return CRNNModelSpec(config or CRNNConfig()).model_summary()


def expected_svtr_weight_path(config: RecognitionModelConfig | None = None) -> Path:
    cfg = config or RecognitionModelConfig()
    return Path(cfg.backbone.pretrained_path)
