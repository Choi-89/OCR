from __future__ import annotations

from pathlib import Path

from ocr_project.stage6_deployment.api_integration import get_health_payload
from ocr_project.stage6_deployment.versioning import (
    RegistryPaths,
    create_version_json,
    deploy_model,
    deployment_history,
    health_payload,
    initialize_model_registry,
    read_active_version,
    register_version,
    rollback_model,
    rollback_triggers,
    version_api_payload,
)


def _paths(root: Path) -> RegistryPaths:
    return RegistryPaths(
        root=root / "backend/ocr",
        registry_dir=root / "backend/ocr/model_registry",
        inference_dir=root / "backend/ocr/inference",
        active_version_file=root / "backend/ocr/active_version.txt",
        registry_index=root / "backend/ocr/registry_index.json",
        deployment_log=root / "backend/ocr/deployment_log.jsonl",
    )


def _make_source(root: Path, marker: str) -> tuple[Path, Path]:
    inference = root / f"source_{marker}" / "inference"
    configs = root / f"source_{marker}" / "configs"
    for model in ("det", "rec", "cls"):
        model_dir = inference / model
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "inference.pdmodel").write_text(marker, encoding="utf-8")
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "det_config.yaml").write_text("model: det\n", encoding="utf-8")
    return inference, configs


def test_register_deploy_and_rollback_version(tmp_path) -> None:
    paths = _paths(tmp_path)
    initialize_model_registry(paths)
    source_100, configs_100 = _make_source(tmp_path, "v100")
    source_110, configs_110 = _make_source(tmp_path, "v110")

    register_version("v1.0.0", source_100, configs_100, paths, create_version_json("v1.0.0", performance={"det_f1": 0.89}))
    register_version("v1.1.0", source_110, configs_110, paths, create_version_json("v1.1.0", performance={"det_f1": 0.91}))
    deployed = deploy_model("v1.0.0", paths)
    upgraded = deploy_model("v1.1.0", paths)
    rolled_back = rollback_model("v1.0.0", paths, reason="test rollback", force=True)

    assert deployed["to"] == "v1.0.0"
    assert upgraded["from"] == "v1.0.0"
    assert rolled_back["to"] == "v1.0.0"
    assert read_active_version(paths.active_version_file) == "v1.0.0"
    assert (paths.inference_dir / "det" / "inference.pdmodel").read_text(encoding="utf-8") == "v100"
    assert len(deployment_history(paths)["history"]) == 6


def test_version_api_and_health_payload(tmp_path) -> None:
    paths = _paths(tmp_path)
    initialize_model_registry(paths)
    source, configs = _make_source(tmp_path, "v100")
    register_version("v1.0.0", source, configs, paths, create_version_json("v1.0.0", performance={"rec_cer": 0.03}))
    deploy_model("v1.0.0", paths)

    version = version_api_payload(paths)
    health = health_payload(paths, error_rate_1h=0.01, avg_response_ms=1200, ocr_enabled=True)

    assert version["version"] == "v1.0.0"
    assert version["performance"]["rec_cer"] == 0.03
    assert health["ocr_version"] == "v1.0.0"
    assert health["error_rate_1h"] == 0.01
    assert get_health_payload()["status"] == "ok"


def test_rollback_triggers() -> None:
    triggers = rollback_triggers(error_rate=0.07, previous_error_rate=0.01, consecutive_exceptions=10, memory_gb=4.5, health_ok=False)

    assert set(triggers) == {"error_rate_exceeded", "consecutive_exceptions", "memory_limit_exceeded", "health_check_failed"}
