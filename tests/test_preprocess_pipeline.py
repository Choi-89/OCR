from __future__ import annotations

from pathlib import Path

import numpy as np

from ocr_project.stage2_preprocess.preprocess import (
    PreprocessPipeline,
    apply_binarization,
    ensure_bgr_uint8,
    estimate_orientation_180,
    normalize_image,
    resize_for_detection,
    resize_for_recognition,
    rotate_quadrant,
)


CONFIG_PATH = Path("C:/OCR/configs/preprocess_config.yaml")


def test_grayscale_to_bgr_conversion() -> None:
    gray = np.full((32, 48), 120, dtype=np.uint8)
    converted = ensure_bgr_uint8(gray)
    assert converted.shape == (32, 48, 3)
    assert converted.dtype == np.uint8


def test_manual_angle_correction_rotates_180() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    rotated = rotate_quadrant(image, 180)
    assert rotated.shape == image.shape


def test_binarization_returns_three_channels() -> None:
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    binary = apply_binarization(image, "adaptive")
    assert binary.shape == image.shape


def test_detection_resize_outputs_32_multiple() -> None:
    image = np.full((777, 1111, 3), 255, dtype=np.uint8)
    from ocr_project.stage2_preprocess.preprocess import ResizeConfig

    resized, padding = resize_for_detection(image, ResizeConfig())
    assert resized.shape[0] % 32 == 0
    assert resized.shape[1] % 32 == 0
    assert len(padding) == 4


def test_recognition_resize_keeps_height_32() -> None:
    image = np.full((128, 512, 3), 255, dtype=np.uint8)
    from ocr_project.stage2_preprocess.preprocess import ResizeConfig

    resized, _ = resize_for_recognition(image, ResizeConfig())
    assert resized.shape[0] == 32
    assert resized.shape[1] == 320


def test_normalize_outputs_float32() -> None:
    image = np.full((32, 64, 3), 255, dtype=np.uint8)
    from ocr_project.stage2_preprocess.preprocess import NormalizeConfig

    normalized = normalize_image(image, NormalizeConfig())
    assert normalized.dtype == np.float32


def test_orientation_estimator_returns_known_quadrant() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[70:95, 10:90] = 0
    assert estimate_orientation_180(image) in {0, 180}


def test_pipeline_run_returns_required_fields() -> None:
    pipeline = PreprocessPipeline(CONFIG_PATH)
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    result = pipeline.run(image, "det", angle=0)
    assert "image" in result
    assert "padding" in result
    assert "deskew_angle" in result


def test_visualize_returns_canvas() -> None:
    pipeline = PreprocessPipeline(CONFIG_PATH)
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    canvas = pipeline.visualize(image, "rec")
    assert canvas.ndim == 3
