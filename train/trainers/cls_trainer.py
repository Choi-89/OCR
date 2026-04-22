from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from train.datasets.cls_dataset import ClsDataset
from train.utils.local_monitor import LocalRunLogger


class ClsTrainer:
    def __init__(self, config_path: str | Path, data_dir: str | Path, experiment_name: str = "cls_debug"):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.experiment_name = experiment_name
        self.local_logger = LocalRunLogger.create(Path.cwd(), "cls", experiment_name, self.config_path)
        self.dataset = ClsDataset(Path(data_dir) / "train", Path("train/configs/preprocess_config.yaml"))

    def build_debug_batch(self) -> dict[str, Any]:
        sample = self.dataset[0]
        return {
            "image_shape": sample.image.shape,
            "label": sample.label,
            "filename": sample.filename,
            "log_dir": str(self.local_logger.root_dir),
        }
