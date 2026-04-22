from __future__ import annotations

from pathlib import Path
from typing import Any

from ocr_project.stage6_deployment.export_model import build_predictor, build_predictor_config


def integration_checklist() -> list[str]:
    """OCR-S02: Checklist for API/service integration."""
    return [
        "create_ocr_service_v2",
        "keep_legacy_ocr_response_contract",
        "load_predictors_once_with_singleton",
        "preserve_enable_ocr_false_mode",
        "validate_box_restore_math",
        "run_api_contract_tests",
        "run_roster_parse_regression",
        "run_load_test",
    ]


def inference_model_dirs(root: str | Path = ".") -> dict[str, str]:
    base = Path(root) / "backend/ocr/inference"
    return {
        "det": str(base / "det"),
        "rec": str(base / "rec"),
        "cls": str(base / "cls"),
    }


def predictor_configs(root: str | Path = ".", use_gpu: bool = False, device_id: int = 0, cpu_threads: int = 4) -> dict[str, dict[str, Any]]:
    return {
        key: build_predictor_config(path, use_gpu=use_gpu, device_id=device_id, cpu_threads=cpu_threads)
        for key, path in inference_model_dirs(root).items()
    }


def create_ocr_predictors(root: str | Path = ".", use_gpu: bool = False, device_id: int = 0, cpu_threads: int = 4) -> dict[str, Any]:
    return {
        key: build_predictor(path, use_gpu=use_gpu, device_id=device_id, cpu_threads=cpu_threads)
        for key, path in inference_model_dirs(root).items()
    }
