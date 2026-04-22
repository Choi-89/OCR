from __future__ import annotations

import json

from ocr_project.stage5_evaluation.quality_gate import (
    check_quality_gate,
    generate_quality_outputs,
    passes_quality_gate,
)


def _det_summary(f1: float = 0.89) -> dict:
    return {
        "metrics": {"precision": 0.88, "recall": 0.87, "f1": f1},
        "error_analysis": {"grid_false_positive_rate": 0.03, "small_cell_miss_rate": 0.12},
    }


def _rec_summary(work_code_accuracy: float = 0.96) -> dict:
    return {
        "metrics": {"cer": 0.028, "wer": 0.04, "accuracy": 0.91},
        "domain_metrics": {"work_code_accuracy": work_code_accuracy, "date_exact_match": 0.95},
        "by_type": {"single_char": {"cer": 0.01}, "handwrite": {"cer": 0.08}},
    }


def _cls_summary() -> dict:
    return {"metrics": {"accuracy": 0.992, "false_positive_rate": 0.003}}


def _e2e_summary() -> dict:
    return {
        "parse_success_rate": 0.94,
        "metrics": {
            "cell_accuracy": 0.912,
            "worker_schedule_accuracy": 0.724,
            "name_accuracy": 0.89,
            "code_distribution_error": 0.038,
        },
        "by_format": [
            {"group": "paper", "cell_accuracy": 0.90},
            {"group": "screen", "cell_accuracy": 0.89},
            {"group": "handwrite", "cell_accuracy": 0.72},
        ],
    }


def _service_summary(cpu_p50_seconds: float = 1.8) -> dict:
    return {
        "speed": {"cpu_p50_seconds": cpu_p50_seconds, "cpu_p95_seconds": 4.2},
        "memory": {"model_loaded_gb": 1.4, "peak_inference_gb": 2.6},
        "stability": {"exceptions_per_100": 0},
        "confidence": {"high_conf_error_rate": 0.01, "low_conf_error_rate": 0.3},
    }


def test_quality_gate_passes_all_stages(tmp_path) -> None:
    result = check_quality_gate(_det_summary(), _rec_summary(), _cls_summary(), _e2e_summary(), _service_summary())
    outputs = generate_quality_outputs(result, tmp_path)

    assert result["final_status"] == "PASS"
    assert result["passed"] is True
    assert (tmp_path / "quality_report.md").exists()
    manifest = json.loads((tmp_path / "deploy_manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_status"] == "PASS"
    assert "quality_report" in outputs


def test_gate1_blocks_e2e_when_detection_fails() -> None:
    result = check_quality_gate(_det_summary(f1=0.7), _rec_summary(), _cls_summary(), _e2e_summary(), _service_summary())

    assert result["final_status"] == "FAIL"
    assert result["gates"][0]["failed_items"] == ["detection"]
    assert result["gates"][1]["skipped"] is True


def test_quality_gate_allows_conditional_pass_for_deployment_only_failure() -> None:
    result = check_quality_gate(_det_summary(), _rec_summary(), _cls_summary(), _e2e_summary(), _service_summary(cpu_p50_seconds=3.5))

    assert result["final_status"] == "CONDITIONAL_PASS"
    assert result["conditional_pass"] is True


def test_missing_service_summary_does_not_pass_gate3() -> None:
    result = check_quality_gate(_det_summary(), _rec_summary(), _cls_summary(), _e2e_summary(), {})

    assert result["final_status"] == "CONDITIONAL_PASS"
    assert result["gates"][2]["failed_items"] == ["deployment"]


def test_backward_compatible_cer_gate() -> None:
    assert passes_quality_gate(0.02) is True
    assert passes_quality_gate(0.04) is False
