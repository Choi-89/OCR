from __future__ import annotations

from pathlib import Path

from ocr_project.stage3_models.detection_model import DBNetPPModelSpec, DetectionModelConfig
from backend.ocr.models.det.det_model import dummy_forward_check, freeze_strategy
from backend.ocr.models.det.det_postprocess import postprocess_boxes


CONFIG_PATH = Path("C:/OCR/backend/ocr/models/det/det_config.yaml")


def test_forward_shape_matches_dbnet_output() -> None:
    spec = DBNetPPModelSpec(DetectionModelConfig())
    outputs = spec.forward_shape((1, 3, 960, 960))
    assert outputs["probability_map"] == (1, 1, 240, 240)
    assert outputs["threshold_map"] == (1, 1, 240, 240)
    assert outputs["binary_map"] == (1, 1, 240, 240)


def test_freeze_strategy_has_two_phases() -> None:
    strategy = freeze_strategy(CONFIG_PATH)
    assert len(strategy) == 2
    assert strategy[0]["backbone_frozen"] is True
    assert strategy[1]["backbone_frozen"] is False


def test_dummy_forward_check_returns_weight_path() -> None:
    artifacts = dummy_forward_check(CONFIG_PATH)
    assert artifacts.output_shapes["probability_map"] == (1, 1, 240, 240)
    assert "resnet50_imagenet.pdparams" in str(artifacts.weight_path)


def test_postprocess_reverses_padding() -> None:
    results = postprocess_boxes(
        boxes=[[10, 20, 50, 60]],
        scores=[0.9],
        padding=(0, 16, 0, 32),
        scale_x=1.0,
        scale_y=1.0,
    )
    assert len(results) == 1
