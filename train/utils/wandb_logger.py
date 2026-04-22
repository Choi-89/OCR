from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WandbRun:
    enabled: bool
    run: Any = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self.run is not None:
            self.run.log(metrics, step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        if self.enabled and self.run is not None:
            self.run.summary.update(values)

    def save(self, path: str) -> None:
        if self.enabled and self.run is not None:
            self.run.save(path)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


def init_wandb_run(
    *,
    project: str,
    experiment_name: str,
    config: dict[str, Any],
    tags: list[str],
    notes: str = "",
    mode: str = "online",
) -> WandbRun:
    if mode == "disabled":
        return WandbRun(enabled=False)
    try:
        import wandb

        run = wandb.init(project=project, name=experiment_name, config=config, tags=tags, notes=notes, mode=mode)
        return WandbRun(enabled=True, run=run)
    except Exception:
        return WandbRun(enabled=False)
