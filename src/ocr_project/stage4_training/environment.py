from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GPUProfile:
    name: str = "NVIDIA GeForce RTX 2080 Ti"
    count: int = 7
    vram_gb: int = 11
    driver_version: str = "580.126.09"
    cuda_version: str = "13.0"


@dataclass(slots=True)
class DataLoaderProfile:
    batch_size: int
    num_workers: int
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    use_lmdb: bool = False


@dataclass(slots=True)
class TrainingEnvironment:
    framework: str = "PaddlePaddle"
    paddle_version: str = "2.6.0"
    paddleocr_version: str = "2.8.0"
    gpu: GPUProfile = field(default_factory=GPUProfile)
    precision: str = "fp16"
    distributed: bool = False
    selected_gpus: tuple[int, ...] = (0,)
    cls_loader: DataLoaderProfile = field(default_factory=lambda: DataLoaderProfile(batch_size=128, num_workers=8))
    det_loader: DataLoaderProfile = field(default_factory=lambda: DataLoaderProfile(batch_size=12, num_workers=6))
    rec_loader: DataLoaderProfile = field(default_factory=lambda: DataLoaderProfile(batch_size=64, num_workers=8))
    notes: tuple[str, ...] = (
        "Use classifier training first as an environment smoke test.",
        "Keep DBNet++ single-GPU batch size conservative on 11GB VRAM.",
        "Enable multi-GPU only after single-step forward and checkpoint flow are stable.",
    )


def build_training_environment() -> TrainingEnvironment:
    """Return the default OCR-T01 environment tuned for RTX 2080 Ti 11GB cards."""
    return TrainingEnvironment()


def initialize_train_workspace(root: str | Path) -> dict[str, str]:
    base = Path(root) / "train"
    directories = {
        "root": base,
        "configs": base / "configs",
        "datasets": base / "datasets",
        "models": base / "models",
        "losses": base / "losses",
        "optimizers": base / "optimizers",
        "trainers": base / "trainers",
        "evaluators": base / "evaluators",
        "utils": base / "utils",
        "scripts": base / "scripts",
        "weights": base / "weights",
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    for subdir in ("det", "rec", "cls"):
        (directories["checkpoints"] / subdir).mkdir(parents=True, exist_ok=True)
    return {name: str(path) for name, path in directories.items()}


def environment_summary() -> dict[str, Any]:
    env = build_training_environment()
    return {
        "framework": env.framework,
        "paddle_version": env.paddle_version,
        "paddleocr_version": env.paddleocr_version,
        "gpu": asdict(env.gpu),
        "precision": env.precision,
        "distributed": env.distributed,
        "selected_gpus": list(env.selected_gpus),
        "cls_loader": asdict(env.cls_loader),
        "det_loader": asdict(env.det_loader),
        "rec_loader": asdict(env.rec_loader),
        "notes": list(env.notes),
    }
