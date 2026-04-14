from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from train.datasets.rec_dataset import RecDataset, rec_collate_fn


class RecTrainer:
    def __init__(self, config_path: str | Path, data_dir: str | Path, dict_path: str | Path):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.dataset = RecDataset(
            Path(data_dir) / "train" / "crop",
            Path(data_dir) / "train" / "rec_gt.txt",
            dict_path,
            Path("train/configs/preprocess_config.yaml"),
            Path("train/configs/augment_config.yaml"),
        )

    def build_debug_batch(self) -> dict[str, Any]:
        batch = rec_collate_fn([self.dataset[0]])
        return {
            "image_shape": batch["image"].shape,
            "label_shape": batch["label"].shape,
            "label_len": batch["label_len"].tolist(),
            "text_type": batch["text_type"],
        }
