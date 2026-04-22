from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SearchSpace:
    fixed: dict[str, Any]
    search: dict[str, list[Any]]
    order: list[str]
    early_stopping: dict[str, Any]
    first_run: dict[str, Any]


@dataclass(slots=True)
class HyperParameterPlan:
    cls: SearchSpace
    det: SearchSpace
    rec: SearchSpace
    experiment_name_rule: str = "{model}_{index:03d}_{field}_{value}"
    seed: int = 42
    reproducibility_tolerance: float = 0.005


@dataclass(slots=True)
class HyperParameters:
    batch_size: int = 12
    epochs: int = 120
    learning_rate: float = 0.001
    min_learning_rate: float = 0.000001
    warmup_steps: int = 500
    weight_decay: float = 0.0001
    precision: str = "fp16"
    seed: int = 42


def default_hyperparameters() -> HyperParameters:
    """Return the conservative DBNet++ starting point for RTX 2080 Ti 11GB."""
    return HyperParameters()


def build_hyperparameter_plan() -> HyperParameterPlan:
    cls = SearchSpace(
        fixed={
            "input_size": [48, 192],
            "model": "MobileNetV3Small",
            "optimizer": "Adam",
            "loss": "FocalLoss(gamma=2, alpha=0.25)",
            "scheduler": "StepDecay(step=10, gamma=0.1)",
            "weight_decay": 1e-4,
            "num_workers": 8,
        },
        search={
            "base_lr": [1e-4, 5e-4, 1e-3, 5e-3],
            "batch_size": [64, 128],
            "epochs": [30, 50, 100],
            "inference_threshold": [0.5, 0.6, 0.7, 0.8],
        },
        order=["base_lr", "batch_size", "epochs", "inference_threshold"],
        early_stopping={"enabled": True, "patience": 10, "monitor": "val_accuracy", "min_delta": 0.001, "restore_best": True},
        first_run={"base_lr": 1e-3, "batch_size": 128, "epochs": 50, "inference_threshold": 0.7},
    )
    det = SearchSpace(
        fixed={
            "model": "DBNet++",
            "backbone": "ResNet50",
            "fpn_out_channels": 256,
            "db_loss_alpha": 10,
            "db_loss_beta": 5,
            "db_k": 50,
            "ohem_ratio": 3,
            "optimizer": "Adam",
            "weight_decay": 1e-4,
            "backbone_lr_ratio": 0.1,
            "freeze_epochs": 5,
            "scheduler": "WarmupCosineDecay",
            "warmup_steps": 500,
            "input_max_side": 960,
        },
        search={
            "base_lr": [5e-4, 1e-3, 3e-3],
            "batch_size": [8, 12, 16],
            "epochs": [100, 200, 300],
            "min_lr": [1e-6, 1e-5],
            "unclip_ratio": [1.3, 1.5, 1.7, 2.0],
            "prob_threshold": [0.2, 0.3, 0.4, 0.5],
            "box_threshold": [0.3, 0.5, 0.7],
            "min_box_size": [2, 3, 5],
        },
        order=["batch_size", "base_lr", "epochs", "min_lr", "unclip_ratio", "prob_threshold", "box_threshold", "min_box_size"],
        early_stopping={"enabled": True, "patience": 20, "monitor": "val_f1", "min_delta": 0.002, "restore_best": True},
        first_run={"base_lr": 1e-3, "min_lr": 1e-6, "batch_size": 12, "epochs": 120, "unclip_ratio": 1.5, "prob_threshold": 0.3, "box_threshold": 0.5, "min_box_size": 3},
    )
    rec = SearchSpace(
        fixed={
            "model": "SVTR-Tiny",
            "input_height": 32,
            "input_max_width": 320,
            "input_min_width": 32,
            "ctc_blank_index": "vocab_size - 1",
            "optimizer": "Adam",
            "weight_decay": 1e-5,
            "gradient_clip": 5.0,
            "freeze_epochs": 5,
            "scheduler": "WarmupCosineDecay",
            "warmup_steps": 300,
            "decoder": "greedy",
        },
        search={
            "base_lr": [5e-4, 1e-3, 3e-3],
            "batch_size": [64, 128],
            "epochs": [50, 100, 150],
            "label_smoothing": [0.0, 0.05, 0.1],
            "min_lr": [1e-6, 1e-5],
            "beam_width": [3, 5, 10],
        },
        order=["batch_size", "base_lr", "epochs", "label_smoothing", "min_lr", "beam_width"],
        early_stopping={"enabled": True, "patience": 15, "monitor": "val_cer", "min_delta": 0.001, "restore_best": True},
        first_run={"base_lr": 1e-3, "min_lr": 1e-6, "batch_size": 64, "epochs": 80, "label_smoothing": 0.0, "decoder": "greedy", "beam_width": 5},
    )
    return HyperParameterPlan(cls=cls, det=det, rec=rec)


def plan_as_dict() -> dict[str, Any]:
    plan = build_hyperparameter_plan()
    return {
        "cls": asdict(plan.cls),
        "det": asdict(plan.det),
        "rec": asdict(plan.rec),
        "experiment_name_rule": plan.experiment_name_rule,
        "seed": plan.seed,
        "reproducibility_tolerance": plan.reproducibility_tolerance,
    }


def initialize_experiment_tables(root: str | Path) -> dict[str, str]:
    base = Path(root) / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    columns = "experiment_name,start_time,end_time,changed_field,changed_value,best_epoch,best_metric,train_loss,val_loss,curve_path,checkpoint_path,notes\n"
    outputs: dict[str, str] = {}
    for model in ("cls", "det", "rec"):
        path = base / f"{model}_results.csv"
        if not path.exists():
            path.write_text(columns, encoding="utf-8")
        outputs[model] = str(path)
    return outputs
