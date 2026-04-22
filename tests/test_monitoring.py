from __future__ import annotations

from pathlib import Path

from ocr_project.stage4_training.monitoring import build_monitoring_plan, initialize_monitoring_workspace
from train.utils.checkpoint import CheckpointManager
from train.utils.health import TrainingHealthState, check_training_health
from train.utils.local_monitor import LocalRunLogger


def test_monitoring_plan_uses_shiftflow_project() -> None:
    plan = build_monitoring_plan()
    assert plan.project == "shiftflow-ocr"
    assert "train/loss_total" in plan.det.step
    assert plan.rec.best_mode == "min"


def test_initialize_monitoring_workspace_creates_logs(tmp_path: Path) -> None:
    paths = initialize_monitoring_workspace(tmp_path, "det", "det_test")
    assert Path(paths["train_log"]).exists()
    assert Path(paths["val_log"]).exists()


def test_health_check_warns_on_large_grad_norm() -> None:
    messages = check_training_health(1.0, 1001.0, 1, TrainingHealthState())
    assert messages and messages[0].startswith("grad_norm_spike")


def test_checkpoint_manager_keeps_best_and_latest(tmp_path: Path) -> None:
    manager = CheckpointManager.create(tmp_path, "val/f1", mode="max", keep_last=1)
    manager.save(1, {"model_state_dict": {"a": 1}}, {"val/f1": 0.5})
    manager.save(2, {"model_state_dict": {"a": 2}}, {"val/f1": 0.6})
    assert (tmp_path / "latest.pdparams").exists()
    assert (tmp_path / "best.pdparams").exists()
    assert len(list(tmp_path.glob("epoch_*.pdparams"))) == 1


def test_local_run_logger_writes_summary(tmp_path: Path) -> None:
    logger = LocalRunLogger.create(tmp_path, "cls", "cls_test")
    summary = logger.write_summary(["experiment: cls_test"])
    assert summary.exists()
