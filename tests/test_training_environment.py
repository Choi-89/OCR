from __future__ import annotations

from pathlib import Path

from ocr_project.stage4_training.environment import build_training_environment, initialize_train_workspace


def test_training_environment_uses_rtx_2080_ti_defaults() -> None:
    env = build_training_environment()
    assert env.gpu.vram_gb == 11
    assert env.det_loader.batch_size == 12
    assert env.rec_loader.batch_size == 64
    assert env.cls_loader.batch_size == 128


def test_initialize_train_workspace_creates_expected_roots(tmp_path: Path) -> None:
    workspace = initialize_train_workspace(tmp_path)
    assert Path(workspace["configs"]).exists()
    assert Path(workspace["checkpoints"]).exists()
    assert Path(workspace["logs"]).exists()
