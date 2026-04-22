from __future__ import annotations

import json

from ocr_project.stage5_evaluation.e2e_metrics import (
    E2EEvaluator,
    ScheduleMatrix,
    compare_schedule,
    parse_schedule_from_tokens,
    run_or_wrap_ocr,
)


def _gt_payload() -> dict:
    return {
        "image_path": "paper_hospital_0001.jpg",
        "format": "paper",
        "industry": "hospital",
        "year_month": "2026-04",
        "workers": ["Kim", "Lee"],
        "schedule": {
            "Kim": {"1": "D", "2": "E"},
            "Lee": {"1": "N", "2": "OFF"},
        },
        "metadata": {"total_cells": 4},
    }


def _ocr_tokens() -> list[dict]:
    return [
        {"text": "1", "bbox": [100, 0, 120, 20]},
        {"text": "2", "bbox": [140, 0, 160, 20]},
        {"text": "Kim", "bbox": [0, 40, 40, 60]},
        {"text": "D", "bbox": [100, 40, 120, 60]},
        {"text": "E", "bbox": [140, 40, 160, 60]},
        {"text": "Lee", "bbox": [0, 80, 40, 100]},
        {"text": "N", "bbox": [100, 80, 120, 100]},
        {"text": "OFF", "bbox": [140, 80, 170, 100]},
    ]


def test_parse_schedule_from_tokens_builds_matrix() -> None:
    gt = ScheduleMatrix(**{key: value for key, value in _gt_payload().items() if key != "metadata"}, metadata={})
    tokens = run_or_wrap_ocr("paper_hospital_0001.jpg", _ocr_tokens())

    pred, parse_success = parse_schedule_from_tokens(tokens, gt, E2EEvaluator("unused").parse_config)

    assert parse_success is True
    assert pred.workers == ["Kim", "Lee"]
    assert pred.schedule["Kim"] == {"1": "D", "2": "E"}
    assert pred.schedule["Lee"] == {"1": "N", "2": "OFF"}


def test_evaluate_dataset_computes_e2e_metrics_and_outputs(tmp_path) -> None:
    gt_dir = tmp_path / "e2e_gt"
    gt_dir.mkdir()
    (gt_dir / "paper_hospital_0001.json").write_text(json.dumps(_gt_payload()), encoding="utf-8")

    evaluator = E2EEvaluator(gt_dir)
    results = evaluator.evaluate_dataset(tmp_path, {"paper_hospital_0001.jpg": _ocr_tokens()})
    outputs = evaluator.generate_report(results, tmp_path / "results")

    assert results["summary"]["parse_success_rate"] == 1.0
    assert results["summary"]["metrics"]["cell_accuracy"] == 1.0
    assert results["summary"]["metrics"]["worker_schedule_accuracy"] == 1.0
    assert results["summary"]["metrics"]["name_accuracy"] == 1.0
    assert results["summary"]["metrics"]["code_distribution_error"] == 0.0
    assert results["summary"]["all_targets_met"] is True
    assert (tmp_path / "results" / "summary.json").exists()
    assert (tmp_path / "results" / "e2e_report.md").exists()
    assert "summary" in outputs


def test_compare_schedule_classifies_missed_cell() -> None:
    gt = ScheduleMatrix(**{key: value for key, value in _gt_payload().items() if key != "metadata"}, metadata={})
    pred = ScheduleMatrix(
        image_path="paper_hospital_0001.jpg",
        format="paper",
        industry="hospital",
        year_month="2026-04",
        workers=["Kim", "Lee"],
        schedule={"Kim": {"1": "D", "2": ""}, "Lee": {"1": "N", "2": "OFF"}},
    )

    result = compare_schedule(gt, pred, parse_success=True)

    assert result.cell_accuracy == 0.75
    assert result.worker_schedule_accuracy == 0.5
    assert result.errors == [{"worker": "Kim", "date": "2", "gt_code": "E", "pred_code": "", "error_type": "det_miss"}]
