from __future__ import annotations

from pathlib import Path

from ocr_project.stage4_training.hyperparameters import build_hyperparameter_plan, initialize_experiment_tables


def test_hyperparameter_plan_has_model_specific_first_runs() -> None:
    plan = build_hyperparameter_plan()
    assert plan.cls.first_run["batch_size"] == 128
    assert plan.det.first_run["batch_size"] == 12
    assert plan.rec.first_run["batch_size"] == 64
    assert plan.det.early_stopping["monitor"] == "val_f1"


def test_initialize_experiment_tables_writes_three_csvs(tmp_path: Path) -> None:
    outputs = initialize_experiment_tables(tmp_path)
    assert set(outputs) == {"cls", "det", "rec"}
    for csv_path in outputs.values():
        assert Path(csv_path).exists()
