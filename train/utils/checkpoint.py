from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def save_checkpoint(output_dir: str | Path, payload: dict[str, Any], filename: str = "latest.pdparams") -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(slots=True)
class CheckpointManager:
    save_dir: Path
    monitor: str
    mode: str = "max"
    keep_last: int = 3
    best_metric: float | None = None

    @classmethod
    def create(
        cls,
        save_dir: str | Path,
        monitor: str,
        *,
        mode: str = "max",
        keep_last: int = 3,
    ) -> "CheckpointManager":
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return cls(save_dir=directory, monitor=monitor, mode=mode, keep_last=keep_last)

    def save(self, epoch: int, state: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
        payload = {
            "epoch": epoch,
            "model_state_dict": state.get("model_state_dict", {}),
            "optimizer_state_dict": state.get("optimizer_state_dict", {}),
            "scheduler_state_dict": state.get("scheduler_state_dict", {}),
            "metrics": metrics,
            "config": state.get("config", {}),
        }
        epoch_path = save_checkpoint(self.save_dir, payload, f"epoch_{epoch:04d}.pdparams")
        latest_path = self.save_dir / "latest.pdparams"
        shutil.copy2(epoch_path, latest_path)

        metric = metrics.get(self.monitor)
        is_best = metric is not None and self._is_better(float(metric))
        best_path = self.save_dir / "best.pdparams"
        if is_best:
            self.best_metric = float(metric)
            shutil.copy2(epoch_path, best_path)

        self.cleanup_old_checkpoints()
        return {
            "epoch_path": str(epoch_path),
            "latest_path": str(latest_path),
            "best_path": str(best_path) if best_path.exists() else "",
            "is_best": is_best,
            "best_metric": self.best_metric,
        }

    def cleanup_old_checkpoints(self) -> None:
        checkpoints = sorted(self.save_dir.glob("epoch_*.pdparams"))
        stale = checkpoints[: max(0, len(checkpoints) - self.keep_last)]
        for path in stale:
            path.unlink(missing_ok=True)

    def _is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric
