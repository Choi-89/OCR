from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


WANDB_PROJECT = "shiftflow-ocr"


@dataclass(slots=True)
class LoggingSpec:
    step: list[str]
    epoch: list[str]
    visualizations: list[str] = field(default_factory=list)
    best_metric: str = ""
    best_mode: str = "max"


@dataclass(slots=True)
class HealthRule:
    name: str
    severity: str
    condition: str
    action: str


@dataclass(slots=True)
class MonitoringPlan:
    project: str
    det: LoggingSpec
    rec: LoggingSpec
    cls: LoggingSpec
    health_rules: list[HealthRule]
    local_log_files: tuple[str, ...] = ("train_log.csv", "val_log.csv", "config.yaml", "summary.txt")
    visualization_interval_epochs: int = 10
    checkpoint_keep_last: int = 3


def build_monitoring_plan() -> MonitoringPlan:
    return MonitoringPlan(
        project=WANDB_PROJECT,
        det=LoggingSpec(
            step=["train/loss_total", "train/loss_prob", "train/loss_thresh", "train/loss_binary", "train/learning_rate", "train/grad_norm"],
            epoch=["val/precision", "val/recall", "val/f1", "val/loss_total", "epoch_time", "best_val_f1"],
            visualizations=["val_bbox_overlay", "probability_heatmap", "threshold_heatmap"],
            best_metric="val/f1",
            best_mode="max",
        ),
        rec=LoggingSpec(
            step=["train/loss_ctc", "train/learning_rate", "train/grad_norm"],
            epoch=["val/cer", "val/wer", "val/accuracy", "val/cer_single_char", "val/cer_date", "val/cer_handwrite", "val/loss_ctc", "best_val_cer"],
            visualizations=["crop_prediction_table", "top_cer_errors"],
            best_metric="val/cer",
            best_mode="min",
        ),
        cls=LoggingSpec(
            step=["train/loss_focal", "train/learning_rate"],
            epoch=["val/accuracy", "val/false_positive_rate", "val/false_negative_rate", "val/loss_focal"],
            best_metric="val/accuracy",
            best_mode="max",
        ),
        health_rules=[
            HealthRule("loss_nan", "stop", "loss is NaN", "raise RuntimeError"),
            HealthRule("loss_diverging", "stop", "loss increases for 10 consecutive logged steps", "stop and lower learning rate"),
            HealthRule("grad_norm_spike", "warn", "grad_norm > 1000", "warn and inspect gradient clipping"),
            HealthRule("val_loss_gap", "warn", "val_loss > train_loss * 10 after epoch 5", "inspect overfitting or leakage"),
            HealthRule("metric_plateau", "watch", "no validation improvement before early stopping", "review next epoch"),
        ],
    )


def monitoring_targets() -> list[str]:
    """OCR-T04: Signals to monitor during training."""
    plan = build_monitoring_plan()
    return [
        plan.project,
        "wandb",
        "local_csv_logs",
        "checkpoint_manager",
        "health_checks",
        "prediction_visualizations",
        "completion_notifications",
    ]


def monitoring_plan_as_dict() -> dict[str, Any]:
    plan = build_monitoring_plan()
    return {
        "project": plan.project,
        "det": asdict(plan.det),
        "rec": asdict(plan.rec),
        "cls": asdict(plan.cls),
        "health_rules": [asdict(rule) for rule in plan.health_rules],
        "local_log_files": list(plan.local_log_files),
        "visualization_interval_epochs": plan.visualization_interval_epochs,
        "checkpoint_keep_last": plan.checkpoint_keep_last,
    }


def initialize_monitoring_workspace(root: str | Path, model: str, experiment_name: str) -> dict[str, str]:
    log_dir = Path(root) / "train" / "logs" / model / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "log_dir": log_dir,
        "train_log": log_dir / "train_log.csv",
        "val_log": log_dir / "val_log.csv",
        "config_snapshot": log_dir / "config.yaml",
        "summary": log_dir / "summary.txt",
        "loss_curve": log_dir / "loss_curve.png",
        "metric_curve": log_dir / "metric_curve.png",
        "lr_curve": log_dir / "lr_curve.png",
    }
    if not paths["train_log"].exists():
        paths["train_log"].write_text("step,epoch,loss_total,loss_prob,loss_thresh,loss_binary,lr,grad_norm,timestamp\n", encoding="utf-8")
    if not paths["val_log"].exists():
        paths["val_log"].write_text("epoch,val_primary_metric,val_loss,epoch_time,is_best,timestamp\n", encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}
