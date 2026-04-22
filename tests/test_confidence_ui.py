from __future__ import annotations

import json

from ocr_project.stage6_deployment.confidence_ui import (
    ConfidenceConfig,
    FeedbackStoreConfig,
    OCRFeedbackRequest,
    build_cell_id,
    build_feedback_dataset,
    confidence_level,
    confidence_summary,
    enrich_ocr_response,
    feedback_stats,
    submit_feedback,
)


def test_confidence_levels_and_summary() -> None:
    cfg = ConfidenceConfig(high_threshold=0.9, low_threshold=0.7)

    assert confidence_level(0.95, cfg) == "high"
    assert confidence_level(0.80, cfg) == "mid"
    assert confidence_level(0.69, cfg) == "low"
    assert confidence_summary([{"confidence_level": "high"}, {"confidence_level": "mid"}, {"confidence_level": "low"}]) == {
        "total_cells": 3,
        "high_confidence": 1,
        "mid_confidence": 1,
        "low_confidence": 1,
        "review_required": 2,
    }


def test_enrich_ocr_response_adds_cell_id_level_and_summary() -> None:
    response = enrich_ocr_response(
        {"results": [{"text": "D", "confidence": 0.93, "bbox": [1, 2, 3, 4]}, {"text": "N", "confidence": 0.5, "bbox": [5, 6, 7, 8]}]},
        import_id="imp_1",
    )

    assert response["results"][0]["confidence_level"] == "high"
    assert response["results"][0]["cell_id"] == "imp_1_cell001"
    assert response["results"][1]["confidence_level"] == "low"
    assert response["summary"]["review_required"] == 1
    assert build_cell_id("imp", 1, row=3, col=7) == "imp_row03_col07"


def test_feedback_submission_stats_and_dataset_build(tmp_path) -> None:
    crop_source = tmp_path / "cell.png"
    crop_source.write_bytes(b"fake-image")
    store = FeedbackStoreConfig(root=tmp_path / "feedback")
    request = OCRFeedbackRequest(
        import_id="imp_1",
        cell_id="imp_1_cell001",
        original_text="O",
        corrected_text="0",
        confidence=0.61,
        confidence_level="low",
        image_crop_path=str(crop_source),
    )

    result = submit_feedback(request, store_config=store, worker_id="worker_1")
    stats = feedback_stats(store.root / store.log_name, retrain_threshold=1)
    dataset_stats = build_feedback_dataset(
        store.root / store.log_name,
        store.root / store.crops_dir_name,
        tmp_path / "feedback_dataset",
        min_feedback_count=1,
        dictionary={"0"},
    )

    assert result["status"] == "recorded"
    assert stats["total_feedback"] == 1
    assert stats["usable_feedback"] == 1
    assert stats["ready_for_retrain"] is True
    assert dataset_stats["created"] is True
    assert (tmp_path / "feedback_dataset" / "rec_gt.txt").read_text(encoding="utf-8").strip().endswith(" 0")


def test_feedback_dataset_waits_for_minimum_count(tmp_path) -> None:
    log = tmp_path / "feedback_log.jsonl"
    log.write_text(json.dumps({"original_text": "A", "corrected_text": "B", "crop_saved": True}) + "\n", encoding="utf-8")

    stats = build_feedback_dataset(log, tmp_path / "crops", tmp_path / "out", min_feedback_count=2)

    assert stats["created"] is False
    assert stats["reason"] == "not_enough_feedback"
