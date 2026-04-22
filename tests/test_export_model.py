from __future__ import annotations

import json

from ocr_project.stage6_deployment.export_model import (
    InferenceExportSpec,
    build_predictor_config,
    default_export_specs,
    export_inference_model,
    initialize_inference_workspace,
    measure_latency,
    metric_delta_report,
    model_equivalence_report,
    verify_export_outputs,
)
from ocr_project.stage6_deployment.service_integration import inference_model_dirs, predictor_configs


def test_default_export_specs_match_s01_input_shapes(tmp_path) -> None:
    specs = default_export_specs(tmp_path)

    assert specs["det"].input_shape == [None, 3, None, None]
    assert specs["rec"].input_shape == [None, 3, 32, None]
    assert specs["cls"].input_shape == [None, 3, 48, 192]


def test_dry_run_export_writes_model_info_and_copies_dict(tmp_path) -> None:
    dict_path = tmp_path / "dict_latest.txt"
    dict_path.write_text("D\n<blank>\n", encoding="utf-8")
    spec = InferenceExportSpec(
        model_key="rec",
        model_name="SVTR-Tiny",
        config_path=tmp_path / "rec_config.yaml",
        checkpoint_path=tmp_path / "best.pdparams",
        output_dir=tmp_path / "backend/ocr/inference/rec",
        input_shape=[None, 3, 32, None],
        dict_path=dict_path,
    )

    result = export_inference_model(spec, dry_run=True)
    info = json.loads(result.model_info_path.read_text(encoding="utf-8"))

    assert result.dry_run is True
    assert (spec.output_dir / "dict.txt").read_text(encoding="utf-8") == "D\n<blank>\n"
    assert info["model_key"] == "rec"
    assert info["input_spec"]["shape"] == [None, 3, 32, None]
    assert verify_export_outputs(spec.output_dir, require_binary=False)["passed"] is True


def test_initialize_inference_workspace_creates_model_dirs(tmp_path) -> None:
    paths = initialize_inference_workspace(tmp_path)

    assert set(paths) == {"det", "rec", "cls"}
    assert (tmp_path / "backend/ocr/inference/det").exists()


def test_predictor_config_points_to_paddle_inference_files(tmp_path) -> None:
    config = build_predictor_config(tmp_path / "det", use_gpu=True, device_id=1, cpu_threads=8)

    assert config["model_file"].endswith("inference.pdmodel")
    assert config["params_file"].endswith("inference.pdiparams")
    assert config["use_gpu"] is True
    assert config["device_id"] == 1
    assert config["cpu_threads"] == 8


def test_equivalence_and_metric_delta_reports() -> None:
    assert model_equivalence_report(1e-6)["passed"] is True
    assert model_equivalence_report(1e-4)["passed"] is False
    assert metric_delta_report(0.89, 0.887, tolerance=0.005)["passed"] is True


def test_measure_latency_returns_percentiles() -> None:
    report = measure_latency(lambda: None, warmup=1, repeat=5)

    assert set(report) == {"p50", "p95", "p99", "mean", "max"}
    assert report["max"] >= report["p50"]


def test_service_integration_predictor_config_map(tmp_path) -> None:
    dirs = inference_model_dirs(tmp_path)
    configs = predictor_configs(tmp_path, use_gpu=False)

    assert set(dirs) == {"det", "rec", "cls"}
    assert configs["rec"]["model_file"].endswith("backend\\ocr\\inference\\rec\\inference.pdmodel") or configs["rec"]["model_file"].endswith("backend/ocr/inference/rec/inference.pdmodel")
