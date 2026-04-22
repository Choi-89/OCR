from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LocalRunLogger:
    root_dir: Path

    @classmethod
    def create(cls, root: str | Path, model: str, experiment_name: str, config_path: str | Path | None = None) -> "LocalRunLogger":
        directory = Path(root) / "train" / "logs" / model / experiment_name
        directory.mkdir(parents=True, exist_ok=True)
        logger = cls(directory)
        logger.ensure_csv("train_log.csv", ["step", "epoch", "loss_total", "loss_prob", "loss_thresh", "loss_binary", "lr", "grad_norm", "timestamp"])
        logger.ensure_csv("val_log.csv", ["epoch", "val_primary_metric", "val_loss", "epoch_time", "is_best", "timestamp"])
        if config_path is not None and Path(config_path).exists():
            shutil.copy2(config_path, directory / "config.yaml")
        return logger

    def ensure_csv(self, filename: str, columns: list[str]) -> Path:
        path = self.root_dir / filename
        if not path.exists():
            path.write_text(",".join(columns) + "\n", encoding="utf-8")
        return path

    def append_train(self, row: dict[str, Any]) -> None:
        self._append("train_log.csv", row)

    def append_val(self, row: dict[str, Any]) -> None:
        self._append("val_log.csv", row)

    def write_summary(self, lines: list[str]) -> Path:
        path = self.root_dir / "summary.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _append(self, filename: str, row: dict[str, Any]) -> None:
        path = self.root_dir / filename
        if "timestamp" not in row:
            row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        normalized = {column: row.get(column, "") for column in header}
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writerow(normalized)
