from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(slots=True)
class ExperimentLogRow:
    experiment_name: str
    start_time: str
    end_time: str
    changed_field: str
    changed_value: str
    best_epoch: str = ""
    best_metric: str = ""
    train_loss: str = ""
    val_loss: str = ""
    curve_path: str = ""
    checkpoint_path: str = ""
    notes: str = ""


def append_experiment_row(csv_path: str | Path, row: ExperimentLogRow) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(row))
