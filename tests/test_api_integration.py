from __future__ import annotations

from ocr_project.stage6_deployment.api_integration import get_feedback_stats, normalize_ocr_api_response, parse_roster_payload, submit_ocr_feedback
from ocr_project.stage6_deployment.confidence_ui import FeedbackStoreConfig


def test_normalize_ocr_api_response_preserves_legacy_contract() -> None:
    response = normalize_ocr_api_response(
        {
            "results": [{"text": "D", "confidence": 0.98, "bbox": [1, 2, 3, 4], "det_score": 0.9}],
            "processing_time": 1.23,
            "enabled": True,
            "angle_corrected": False,
            "scores": [0.98],
        }
    )

    assert response["results"][0]["text"] == "D"
    assert response["results"][0]["confidence"] == 0.98
    assert response["results"][0]["confidence_level"] == "high"
    assert response["results"][0]["cell_id"] == "imp_pending_cell001"
    assert response["summary"]["high_confidence"] == 1


def test_parse_roster_payload_wraps_ocr_results_as_entries() -> None:
    payload = parse_roster_payload({"results": [{"text": "N", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]})

    assert payload["entries"][0]["text"] == "N"
    assert payload["ocr"]["results"] == payload["entries"]


def test_feedback_api_helpers_record_and_report_stats(tmp_path) -> None:
    store = FeedbackStoreConfig(root=tmp_path / "feedback")
    result = submit_ocr_feedback(
        {
            "import_id": "imp_1",
            "cell_id": "imp_1_cell001",
            "original_text": "D",
            "corrected_text": "O",
            "confidence": 0.55,
            "confidence_level": "low",
        },
        store_config=store,
        worker_id="worker_1",
    )
    stats = get_feedback_stats(store, retrain_threshold=10)

    assert result["status"] == "recorded"
    assert stats["total_feedback"] == 1
    assert stats["by_confidence_level"]["low"] == 1
