from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GateCriterion:
    name: str
    value: float
    threshold: float
    op: str
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold if self.op == ">=" else self.value <= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "op": self.op,
            "required": self.required,
            "passed": self.passed,
        }


@dataclass(slots=True)
class GateSection:
    name: str
    criteria: list[GateCriterion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.criteria if item.required)

    @property
    def conditional(self) -> bool:
        failed_optional = any(not item.passed for item in self.criteria if not item.required)
        return self.passed and failed_optional

    def failed_items(self) -> list[str]:
        return [item.name for item in self.criteria if item.required and not item.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "conditional": self.conditional,
            "failed_items": self.failed_items(),
            "criteria": [item.as_dict() for item in self.criteria],
            "notes": self.notes,
        }


def passes_quality_gate(cer: float, threshold: float = 0.03) -> bool:
    """Backward-compatible helper used by the project scaffold summary."""
    return cer <= threshold


def check_quality_gate(
    det_summary: dict[str, Any],
    rec_summary: dict[str, Any],
    cls_summary: dict[str, Any],
    e2e_summary: dict[str, Any],
    service_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate1 = check_gate1(det_summary, rec_summary, cls_summary)
    gate2 = check_gate2(e2e_summary) if gate1["passed"] else skipped_gate("Gate 2 - E2E", "Gate 1 failed")
    gate3 = check_gate3(service_summary or {}) if gate2["passed"] else skipped_gate("Gate 3 - Deployment", "Gate 2 failed")
    gates = [gate1, gate2, gate3]
    final_status = "PASS" if all(gate["passed"] for gate in gates) else "FAIL"
    if final_status == "FAIL" and qualifies_conditional_pass(gates):
        final_status = "CONDITIONAL_PASS"
    return {
        "eval_date": str(date.today()),
        "final_status": final_status,
        "passed": final_status == "PASS",
        "conditional_pass": final_status == "CONDITIONAL_PASS",
        "gates": gates,
        "recommended_actions": recommended_actions(gates),
    }


def check_gate1(det_summary: dict[str, Any], rec_summary: dict[str, Any], cls_summary: dict[str, Any]) -> dict[str, Any]:
    sections = {
        "detection": check_detection_gate(det_summary).as_dict(),
        "recognition": check_recognition_gate(rec_summary).as_dict(),
        "classifier": check_classifier_gate(cls_summary).as_dict(),
    }
    return {
        "gate": "Gate 1 - Model",
        "passed": all(section["passed"] for section in sections.values()),
        "details": sections,
        "failed_items": [name for name, section in sections.items() if not section["passed"]],
    }


def check_detection_gate(summary: dict[str, Any]) -> GateSection:
    metrics = summary.get("metrics", {})
    analysis = summary.get("error_analysis", summary.get("error_rates", {}))
    return GateSection(
        name="Detection",
        criteria=[
            GateCriterion("test_f1", metric(metrics, "f1"), 0.82, ">="),
            GateCriterion("test_recall", metric(metrics, "recall"), 0.80, ">="),
            GateCriterion("test_precision", metric(metrics, "precision"), 0.80, ">="),
            GateCriterion("grid_false_positive_rate", metric(analysis, "grid_false_positive_rate"), 0.05, "<=", required=False),
            GateCriterion("small_cell_miss_rate", metric(analysis, "small_cell_miss_rate"), 0.20, "<=", required=False),
        ],
    )


def check_recognition_gate(summary: dict[str, Any]) -> GateSection:
    metrics = summary.get("metrics", {})
    domain = summary.get("domain_metrics", {})
    by_type = summary.get("by_type", {})
    return GateSection(
        name="Recognition",
        criteria=[
            GateCriterion("test_cer", metric(metrics, "cer"), 0.05, "<="),
            GateCriterion("test_wer", metric(metrics, "wer"), 0.08, "<="),
            GateCriterion("test_accuracy", metric(metrics, "accuracy"), 0.85, ">="),
            GateCriterion("work_code_accuracy", metric(domain, "work_code_accuracy"), 0.93, ">="),
            GateCriterion("date_exact_match", metric(domain, "date_exact_match"), 0.90, ">="),
            GateCriterion("single_char_cer", nested_metric(by_type, "single_char", "cer"), 0.03, "<="),
            GateCriterion("handwrite_cer", nested_metric(by_type, "handwrite", "cer"), 0.12, "<=", required=False),
        ],
    )


def check_classifier_gate(summary: dict[str, Any]) -> GateSection:
    metrics = summary.get("metrics", summary)
    return GateSection(
        name="Angle Classifier",
        criteria=[
            GateCriterion("test_accuracy", metric(metrics, "accuracy"), 0.98, ">="),
            GateCriterion("false_positive_rate", metric(metrics, "false_positive_rate"), 0.01, "<="),
        ],
    )


def check_gate2(e2e_summary: dict[str, Any]) -> dict[str, Any]:
    section = check_e2e_gate(e2e_summary)
    return {
        "gate": "Gate 2 - E2E",
        "passed": section.passed,
        "details": {"e2e": section.as_dict()},
        "failed_items": [] if section.passed else ["e2e"],
    }


def check_e2e_gate(summary: dict[str, Any]) -> GateSection:
    metrics = summary.get("metrics", {})
    criteria = [
        GateCriterion("cell_accuracy", metric(metrics, "cell_accuracy"), 0.88, ">="),
        GateCriterion("worker_schedule_accuracy", metric(metrics, "worker_schedule_accuracy"), 0.65, ">="),
        GateCriterion("name_accuracy", metric(metrics, "name_accuracy"), 0.82, ">="),
        GateCriterion("code_distribution_error", metric(metrics, "code_distribution_error"), 0.08, "<="),
        GateCriterion("parse_success_rate", metric(summary, "parse_success_rate"), 0.90, ">="),
    ]
    criteria.extend(format_criteria(summary.get("by_format", [])))
    return GateSection(name="E2E", criteria=criteria)


def format_criteria(rows: list[dict[str, Any]]) -> list[GateCriterion]:
    thresholds = {"paper": 0.88, "scan": 0.88, "screen": 0.88, "excel": 0.90, "handwrite": 0.70}
    output: list[GateCriterion] = []
    for row in rows:
        name = str(row.get("group", row.get("format", "")))
        if name in thresholds:
            output.append(GateCriterion(f"format_{name}_cell_accuracy", metric(row, "cell_accuracy"), thresholds[name], ">="))
    return output


def check_gate3(service_summary: dict[str, Any]) -> dict[str, Any]:
    section = check_deployment_gate(service_summary)
    return {
        "gate": "Gate 3 - Deployment",
        "passed": section.passed,
        "details": {"deployment": section.as_dict()},
        "failed_items": [] if section.passed else ["deployment"],
    }


def check_deployment_gate(summary: dict[str, Any]) -> GateSection:
    speed = summary.get("speed", {})
    memory = summary.get("memory", {})
    stability = summary.get("stability", {})
    confidence = summary.get("confidence", {})
    return GateSection(
        name="Deployment",
        criteria=[
            GateCriterion("cpu_p50_seconds", metric(speed, "cpu_p50_seconds", 999.0), 3.0, "<="),
            GateCriterion("cpu_p95_seconds", metric(speed, "cpu_p95_seconds", 999.0), 10.0, "<=", required=False),
            GateCriterion("model_loaded_gb", metric(memory, "model_loaded_gb", 999.0), 2.0, "<="),
            GateCriterion("peak_inference_gb", metric(memory, "peak_inference_gb", 999.0), 4.0, "<="),
            GateCriterion("exceptions_per_100", metric(stability, "exceptions_per_100", 999.0), 0.0, "<="),
            GateCriterion("high_conf_error_rate", metric(confidence, "high_conf_error_rate", 1.0), 0.02, "<=", required=False),
            GateCriterion("low_conf_error_rate", metric(confidence, "low_conf_error_rate", 0.0), 0.20, ">=", required=False),
        ],
        notes=["Missing required service summary fields fail Gate 3 until speed, memory, and stability are measured."],
    )


def skipped_gate(name: str, reason: str) -> dict[str, Any]:
    return {"gate": name, "passed": False, "skipped": True, "reason": reason, "details": {}, "failed_items": [name]}


def qualifies_conditional_pass(gates: list[dict[str, Any]]) -> bool:
    gate1 = next((gate for gate in gates if gate["gate"].startswith("Gate 1")), {})
    gate2 = next((gate for gate in gates if gate["gate"].startswith("Gate 2")), {})
    gate3 = next((gate for gate in gates if gate["gate"].startswith("Gate 3")), {})
    return bool(gate1.get("passed") and gate2.get("passed") and not gate3.get("passed"))


def recommended_actions(gates: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for gate in gates:
        for failed in gate.get("failed_items", []):
            if failed == "detection":
                actions.append("Re-tune Detection postprocess thresholds before retraining.")
            elif failed == "recognition":
                actions.append("Check work-code coverage and single-char oversampling before retraining Recognition.")
            elif failed == "classifier":
                actions.append("Increase Angle Classifier threshold conservatism and review false positives.")
            elif failed == "e2e":
                actions.append("Inspect parse_error ratio and roster parser failures before model retraining.")
            elif failed == "deployment":
                actions.append("Optimize input size, backbone, or runtime conversion before service release.")
    return actions


def generate_quality_outputs(
    gate_result: dict[str, Any],
    output_dir: str | Path,
    version: str = "v1.0.0",
    approved_by: str = "OCR team",
    checkpoint_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    quality_report = output / "quality_report.md"
    manifest = output / "deploy_manifest.json"
    gate_summary = output / "gate_summary.json"
    quality_report.write_text(build_quality_report(gate_result, version), encoding="utf-8")
    manifest.write_text(
        json.dumps(build_deploy_manifest(gate_result, version, approved_by, checkpoint_paths or {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    gate_summary.write_text(json.dumps(gate_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"quality_report": str(quality_report), "deploy_manifest": str(manifest), "gate_summary": str(gate_summary)}


def build_quality_report(gate_result: dict[str, Any], version: str = "v1.0.0") -> str:
    lines = [
        "# ShiftFlow OCR Quality Report",
        "",
        f"- version: {version}",
        f"- eval_date: {gate_result['eval_date']}",
        f"- final_status: {gate_result['final_status']}",
        "",
        "## Gate Summary",
        "| gate | result | failed_items |",
        "|---|---|---|",
    ]
    for gate in gate_result["gates"]:
        result = "PASS" if gate.get("passed") else "FAIL"
        lines.append(f"| {gate['gate']} | {result} | {', '.join(gate.get('failed_items', []))} |")
    lines.extend(["", "## Recommended Actions"])
    if gate_result["recommended_actions"]:
        lines.extend(f"- {action}" for action in gate_result["recommended_actions"])
    else:
        lines.append("- Approved for OCR-S01 model conversion.")
    return "\n".join(lines) + "\n"


def build_deploy_manifest(
    gate_result: dict[str, Any],
    version: str,
    approved_by: str,
    checkpoint_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "version": version,
        "approved_date": gate_result["eval_date"],
        "approved_by": approved_by,
        "final_status": gate_result["final_status"],
        "models": {
            "detection": {"checkpoint": checkpoint_paths.get("detection", "train/checkpoints/det/best.pdparams")},
            "recognition": {"checkpoint": checkpoint_paths.get("recognition", "train/checkpoints/rec/best.pdparams")},
            "classifier": {"checkpoint": checkpoint_paths.get("classifier", "train/checkpoints/cls/best.pdparams")},
        },
        "gates": gate_result["gates"],
        "deployment_notes": "",
        "next_version_plan": "",
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def metric(container: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = container.get(key, default)
    return float(value) if value is not None else default


def nested_metric(container: dict[str, Any], outer: str, inner: str, default: float = 0.0) -> float:
    return metric(container.get(outer, {}), inner, default)
