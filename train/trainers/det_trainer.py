from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from train.datasets.det_dataset import DetDataset


class DetTrainer:
    def __init__(self, config_path: str | Path, data_dir: str | Path):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.dataset = DetDataset(
            Path(data_dir) / "train" / "images",
            Path(data_dir) / "train" / "det_gt.txt",
            Path("train/configs/preprocess_config.yaml"),
            Path("train/configs/augment_config.yaml"),
        )

    def build_debug_batch(self) -> dict[str, Any]:
        sample = self.dataset[0]
        return {
            "image_shape": sample.image.shape,
            "prob_map_shape": sample.prob_map.shape,
            "threshold_map_shape": sample.threshold_map.shape,
            "valid_mask_shape": sample.valid_mask.shape,
            "filename": sample.filename,
        }
