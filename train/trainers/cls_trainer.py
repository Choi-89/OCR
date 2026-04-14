from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from train.datasets.cls_dataset import ClsDataset


class ClsTrainer:
    def __init__(self, config_path: str | Path, data_dir: str | Path):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.dataset = ClsDataset(Path(data_dir) / "train", Path("train/configs/preprocess_config.yaml"))

    def build_debug_batch(self) -> dict[str, Any]:
        sample = self.dataset[0]
        return {
            "image_shape": sample.image.shape,
            "label": sample.label,
            "filename": sample.filename,
        }
