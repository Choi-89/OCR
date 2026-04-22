from __future__ import annotations

import numpy as np

from ocr_project.stage5_evaluation.detection_metrics import (
    DetectionEvaluator,
    bbox_iou,
    greedy_match,
)


def test_bbox_iou_handles_perfect_overlap() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_bbox_iou_handles_no_overlap() -> None:
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_greedy_match_is_one_to_one() -> None:
    match = greedy_match(
        [[0, 0, 10, 10], [0, 0, 10, 10]],
        [[0, 0, 10, 10]],
        [False],
        iou_threshold=0.5,
    )
    assert match.tp == 1
    assert match.fp == 1
    assert match.fn == 0


def test_difficult_unmatched_box_is_not_fn_when_ignored() -> None:
    match = greedy_match([], [[0, 0, 10, 10]], [True], iou_threshold=0.5, ignore_difficult=True)
    assert match.fn == 0


def test_detection_evaluator_computes_metrics() -> None:
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    evaluator.update(
        pred_boxes=[[[0, 0, 10, 10], [20, 20, 30, 30]]],
        gt_boxes=[[[0, 0, 10, 10], [40, 40, 50, 50]]],
        difficult_flags=[[False, False]],
        image_paths=["paper_hospital_0001.jpg"],
    )
    metrics = evaluator.compute()
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert round(metrics["f1"], 3) == 0.5


def test_visualize_writes_file(tmp_path) -> None:
    evaluator = DetectionEvaluator()
    target = tmp_path / "viz.png"
    evaluator.visualize(
        np.zeros((64, 64, 3), dtype=np.uint8),
        [[0, 0, 10, 10]],
        [[0, 0, 10, 10]],
        target,
    )
    assert target.exists()
