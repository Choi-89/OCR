from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"active", "superseded", "deprecated", "rollback_candidate"}


@dataclass(slots=True)
class RegistryPaths:
    root: Path = Path("backend/ocr")
    registry_dir: Path = Path("backend/ocr/model_registry")
    inference_dir: Path = Path("backend/ocr/inference")
    active_version_file: Path = Path("backend/ocr/active_version.txt")
    registry_index: Path = Path("backend/ocr/registry_index.json")
    deployment_log: Path = Path("backend/ocr/deployment_log.jsonl")


def release_controls() -> list[str]:
    """OCR-S04: Versioning, rollout, and rollback policies."""
    return ["model_registry", "active_version", "deployment_log", "rollback_policy", "version_api"]


def initialize_model_registry(paths: RegistryPaths | None = None) -> dict[str, str]:
    cfg = paths or RegistryPaths()
    cfg.registry_dir.mkdir(parents=True, exist_ok=True)
    cfg.inference_dir.mkdir(parents=True, exist_ok=True)
    cfg.active_version_file.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.active_version_file.exists():
        cfg.active_version_file.write_text("", encoding="utf-8")
    if not cfg.registry_index.exists():
        write_json(cfg.registry_index, {"versions": [], "active_version": "", "last_updated": today()})
    if not cfg.deployment_log.exists():
        cfg.deployment_log.write_text("", encoding="utf-8")
    return {
        "registry_dir": str(cfg.registry_dir),
        "inference_dir": str(cfg.inference_dir),
        "active_version_file": str(cfg.active_version_file),
        "registry_index": str(cfg.registry_index),
        "deployment_log": str(cfg.deployment_log),
    }


def create_version_json(
    version: str,
    *,
    released_by: str = "OCR team",
    status: str = "rollback_candidate",
    performance: dict[str, float] | None = None,
    changes: str = "",
    known_issues: str = "",
    next_version_plan: str = "",
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    return {
        "version": version,
        "release_date": today(),
        "released_by": released_by,
        "status": status,
        "models": {
            "detection": {"architecture": "DBNet++", "backbone": "ResNet-50", "source_checkpoint": "train/checkpoints/det/best.pdparams"},
            "recognition": {"architecture": "SVTR-Tiny", "dict_size": 11315, "source_checkpoint": "train/checkpoints/rec/best.pdparams"},
            "classifier": {"architecture": "MobileNetV3Small", "source_checkpoint": "train/checkpoints/cls/best.pdparams"},
        },
        "performance": performance or {},
        "gate_result": f"train/quality_gate/{version}/quality_report.md",
        "changes": changes,
        "known_issues": known_issues,
        "next_version_plan": next_version_plan,
    }


def register_version(
    version: str,
    source_inference_dir: str | Path,
    config_dir: str | Path,
    paths: RegistryPaths | None = None,
    version_info: dict[str, Any] | None = None,
) -> Path:
    cfg = paths or RegistryPaths()
    initialize_model_registry(cfg)
    target = cfg.registry_dir / version
    if target.exists():
        raise FileExistsError(f"version already exists: {version}")
    (target / "models").mkdir(parents=True)
    (target / "configs").mkdir(parents=True)
    copy_tree_contents(Path(source_inference_dir), target / "models")
    copy_tree_contents(Path(config_dir), target / "configs")
    info = version_info or create_version_json(version)
    validate_version_json(info)
    write_json(target / "version.json", info)
    upsert_registry_entry(cfg, info, target)
    return target


def deploy_model(version: str, paths: RegistryPaths | None = None, *, by: str = "OCR team") -> dict[str, Any]:
    cfg = paths or RegistryPaths()
    source = cfg.registry_dir / version
    if not source.exists():
        raise FileNotFoundError(f"version not found in registry: {version}")
    info = load_version_json(source)
    validate_version_json(info)
    current = read_active_version(cfg.active_version_file)
    append_deployment_event(cfg.deployment_log, "deploy_start", current, version, by=by)
    replace_inference_dir(source / "models", cfg.inference_dir)
    write_active_version(cfg.active_version_file, version)
    update_registry_status(cfg, current, "superseded")
    update_registry_status(cfg, version, "active")
    append_deployment_event(cfg.deployment_log, "deploy_complete", current, version, by=by)
    return {"from": current, "to": version, "status": "deployed"}


def rollback_model(
    target_version: str,
    paths: RegistryPaths | None = None,
    *,
    reason: str,
    by: str = "OCR team",
    force: bool = False,
) -> dict[str, Any]:
    cfg = paths or RegistryPaths()
    source = cfg.registry_dir / target_version
    if not source.exists():
        raise FileNotFoundError(f"rollback target not found: {target_version}")
    info = load_version_json(source)
    if info.get("status") == "deprecated" and not force:
        raise RuntimeError("target version is deprecated; pass force=True to rollback anyway")
    current = read_active_version(cfg.active_version_file)
    append_deployment_event(cfg.deployment_log, "rollback_start", current, target_version, by=by, reason=reason)
    replace_inference_dir(source / "models", cfg.inference_dir)
    write_active_version(cfg.active_version_file, target_version)
    update_registry_status(cfg, current, "deprecated")
    update_registry_status(cfg, target_version, "active")
    append_deployment_event(cfg.deployment_log, "rollback_complete", current, target_version, by=by, reason=reason)
    return {"from": current, "to": target_version, "status": "rolled_back", "reason": reason}


def replace_inference_dir(source_models_dir: Path, inference_dir: Path) -> None:
    parent = inference_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = parent / f"{inference_dir.name}_new"
    old_dir = parent / f"{inference_dir.name}_old"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if old_dir.exists():
        shutil.rmtree(old_dir)
    shutil.copytree(source_models_dir, temp_dir)
    if inference_dir.exists():
        inference_dir.rename(old_dir)
    temp_dir.rename(inference_dir)
    if old_dir.exists():
        shutil.rmtree(old_dir)


def version_api_payload(paths: RegistryPaths | None = None) -> dict[str, Any]:
    cfg = paths or RegistryPaths()
    active = read_active_version(cfg.active_version_file)
    if not active:
        return {"version": "", "status": "missing"}
    info = load_version_json(cfg.registry_dir / active)
    return {
        "version": active,
        "release_date": info.get("release_date", ""),
        "performance": info.get("performance", {}),
        "status": info.get("status", "active"),
    }


def deployment_history(paths: RegistryPaths | None = None, limit: int = 20) -> dict[str, Any]:
    cfg = paths or RegistryPaths()
    logs = read_deployment_log(cfg.deployment_log, limit=limit)
    return {"history": logs, "active_version": read_active_version(cfg.active_version_file)}


def health_payload(paths: RegistryPaths | None = None, *, error_rate_1h: float = 0.0, avg_response_ms: float = 0.0, ocr_enabled: bool = True) -> dict[str, Any]:
    return {
        "status": "ok",
        "ocr_version": read_active_version((paths or RegistryPaths()).active_version_file),
        "ocr_enabled": ocr_enabled,
        "error_rate_1h": error_rate_1h,
        "avg_response_ms": avg_response_ms,
    }


def rollback_triggers(error_rate: float, previous_error_rate: float, consecutive_exceptions: int, memory_gb: float, health_ok: bool) -> list[str]:
    triggers: list[str] = []
    if error_rate > 0.05 and error_rate >= previous_error_rate * 3:
        triggers.append("error_rate_exceeded")
    if consecutive_exceptions >= 10:
        triggers.append("consecutive_exceptions")
    if memory_gb > 4.0:
        triggers.append("memory_limit_exceeded")
    if not health_ok:
        triggers.append("health_check_failed")
    return triggers


def validate_version_json(info: dict[str, Any]) -> None:
    required = ["version", "release_date", "released_by", "status", "models", "performance"]
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(f"version.json missing fields: {missing}")
    if info["status"] not in VALID_STATUSES:
        raise ValueError(f"invalid version status: {info['status']}")


def load_version_json(version_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(version_dir) / "version.json")


def read_active_version(path: str | Path) -> str:
    source = Path(path)
    return source.read_text(encoding="utf-8").strip() if source.exists() else ""


def write_active_version(path: str | Path, version: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(version, encoding="utf-8")


def upsert_registry_entry(cfg: RegistryPaths, version_info: dict[str, Any], version_path: Path) -> None:
    index = read_json(cfg.registry_index) if cfg.registry_index.exists() else {"versions": [], "active_version": "", "last_updated": today()}
    versions = [entry for entry in index.get("versions", []) if entry.get("version") != version_info["version"]]
    perf = version_info.get("performance", {})
    versions.append(
        {
            "version": version_info["version"],
            "status": version_info["status"],
            "release_date": version_info["release_date"],
            "det_f1": perf.get("det_f1"),
            "rec_cer": perf.get("rec_cer"),
            "path": str(version_path),
        }
    )
    index["versions"] = sorted(versions, key=lambda item: item["version"])
    index["active_version"] = next((entry["version"] for entry in index["versions"] if entry["status"] == "active"), index.get("active_version", ""))
    index["last_updated"] = today()
    write_json(cfg.registry_index, index)


def update_registry_status(cfg: RegistryPaths, version: str, status: str) -> None:
    if not version:
        return
    index = read_json(cfg.registry_index)
    for entry in index.get("versions", []):
        if entry.get("version") == version:
            entry["status"] = status
    if status == "active":
        index["active_version"] = version
    index["last_updated"] = today()
    write_json(cfg.registry_index, index)
    version_json = cfg.registry_dir / version / "version.json"
    if version_json.exists():
        info = read_json(version_json)
        info["status"] = status
        write_json(version_json, info)


def append_deployment_event(path: str | Path, event: str, from_version: str, to_version: str, *, by: str, reason: str | None = None) -> None:
    record = {"event": event, "from": from_version, "to": to_version, "timestamp": now_iso(), "by": by}
    if reason:
        record["reason"] = reason
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_deployment_log(path: str | Path, limit: int = 20) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-limit:]


def copy_tree_contents(source: Path, target: Path) -> None:
    if not source.exists():
        return
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()
