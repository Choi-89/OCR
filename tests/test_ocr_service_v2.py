from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ocr_project.stage6_deployment.ocr_service_v2 import (
    OCRService,
    OCRServiceConfig,
    clip_box,
    config_from_env,
    get_ocr_service,
    make_blank_image,
    parse_angle_output,
    parse_bool,
    reset_ocr_service,
)


ROOT = Path("C:/OCR")


class FakePredictor:
    def __init__(self, output: Any):
        self.output = output
        self.calls = 0

    def run(self, inputs: list[np.ndarray]) -> Any:
        self.calls += 1
        return self.output


def _config(enabled: bool = True) -> OCRServiceConfig:
    return OCRServiceConfig(root=ROOT, enable_ocr=enabled)


def test_disabled_mode_returns_legacy_empty_response() -> None:
    service = OCRService(_config(enabled=False))
    response = service.predict(make_blank_image())

    assert response["enabled"] is False
    assert response["results"] == []


def test_predict_runs_cls_det_rec_pipeline_with_fake_predictors() -> None:
    det = FakePredictor({"boxes": [[10, 10, 60, 40]], "scores": [0.91]})
    rec = FakePredictor({"texts": ["D"], "scores": [0.98]})
    cls = FakePredictor({"angle": 0})
    service = OCRService(_config(), det_predictor=det, rec_predictor=rec, cls_predictor=cls)

    response = service.predict(make_blank_image(width=200, height=200))

    assert response["enabled"] is True
    assert response["results"][0]["text"] == "D"
    assert response["results"][0]["confidence"] == 0.98
    assert response["texts"] == ["D"]
    assert len(response["boxes"]) == len(response["scores"]) == 1
    assert det.calls == 1
    assert rec.calls == 1
    assert cls.calls == 1


def test_restore_boxes_reverses_padding_and_scale() -> None:
    service = OCRService(_config(enabled=False))
    preprocess_result = {
        "stages": {"resized": np.zeros((736, 960, 3), dtype=np.uint8)},
        "padding": (0, 16, 0, 0),
    }

    restored = service._restore_boxes([[100, 50, 200, 80]], preprocess_result, (900, 1200))

    assert restored == [[125, 61, 250, 98]]


def test_angle_parser_and_rotation() -> None:
    assert parse_angle_output({"angle": 180}) == 180
    assert parse_angle_output({"label": "0"}) == 0
    assert parse_angle_output([[0.1, 0.9]]) == 180


def test_singleton_can_be_reset() -> None:
    reset_ocr_service()
    first = get_ocr_service(_config(enabled=False))
    second = get_ocr_service(_config(enabled=False))
    reset_ocr_service()

    assert first is second


def test_env_and_box_helpers(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_OCR", "false")
    monkeypatch.setenv("ENABLE_GPU", "true")
    cfg = config_from_env(ROOT)

    assert cfg.enable_ocr is False
    assert cfg.use_gpu is True
    assert parse_bool("off") is False
    assert clip_box([-1, 5, 300, 2], width=200, height=100) == [0, 2, 200, 5]
