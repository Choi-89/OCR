from __future__ import annotations

from pathlib import Path

import numpy as np

from ocr_project.stage2_preprocess.augmentation import (
    AugmentPipeline,
    clip_bboxes,
    horizontal_flip_with_bboxes,
    random_crop_with_bboxes,
    resize_jitter_with_bboxes,
)


CONFIG_PATH = Path("C:/OCR/configs/augment_config.yaml")


def test_horizontal_flip_updates_bbox_coordinates() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    _, boxes = horizontal_flip_with_bboxes(image, [[10, 10, 50, 40]])
    assert boxes[0] == [150, 10, 190, 40]


def test_random_crop_keeps_list_result() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(42)
    _, boxes, _ = random_crop_with_bboxes(image, [[10, 10, 40, 40], [80, 80, 95, 95]], 0.8, 0.5, rng)
    assert isinstance(boxes, list)


def test_resize_jitter_scales_boxes() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    _, boxes = resize_jitter_with_bboxes(image, [[10, 10, 50, 40]], 0.5)
    assert boxes[0][0] == 5


def test_clip_bboxes_removes_invalid_boxes() -> None:
    boxes = clip_bboxes([[0, 0, 10, 10], [5, 5, 5, 7]], 20, 20)
    assert boxes == [[0, 0, 10, 10]]


def test_run_det_returns_required_fields() -> None:
    pipeline = AugmentPipeline(CONFIG_PATH)
    image = np.full((128, 256, 3), 255, dtype=np.uint8)
    result = pipeline.run_det(image, [[10, 10, 50, 40]], seed=1)
    assert "image" in result
    assert "bboxes" in result
    assert "applied" in result


def test_run_rec_supports_all_text_types() -> None:
    pipeline = AugmentPipeline(CONFIG_PATH)
    image = np.full((32, 128, 3), 255, dtype=np.uint8)
    for text_type in ("single_char", "date", "handwrite", "normal"):
        result = pipeline.run_rec(image, text_type, seed=1)
        assert "image" in result
        assert "applied" in result


def test_visualize_helpers_return_canvas() -> None:
    pipeline = AugmentPipeline(CONFIG_PATH)
    det_canvas = pipeline.visualize_det(np.full((128, 256, 3), 255, dtype=np.uint8), [[10, 10, 50, 40]], 3)
    rec_canvas = pipeline.visualize_rec(np.full((32, 128, 3), 255, dtype=np.uint8), "normal", 3)
    assert det_canvas.ndim == 3
    assert rec_canvas.ndim == 3
