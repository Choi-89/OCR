from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_project.stage2_preprocess.preprocess import PreprocessPipeline

from train.datasets.common import hwc_to_chw, load_image


@dataclass(slots=True)
class ClsSample:
    image: np.ndarray
    label: int
    filename: str


class ClsDataset:
    def __init__(self, data_dir: str | Path, preprocess_config: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.preprocess = PreprocessPipeline(preprocess_config)
        self.samples = []
        for label_dir in ("0", "1"):
            class_root = self.data_dir / label_dir
            if not class_root.exists():
                continue
            for image_path in sorted(path for path in class_root.glob("*.*") if path.is_file()):
                self.samples.append((image_path, int(label_dir)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ClsSample:
        image_path, label = self.samples[index]
        image = load_image(image_path)
        processed = self.preprocess.run(image, mode="det", angle=0)
        resized = processed["stages"]["resized"]
        return ClsSample(
            image=hwc_to_chw(resized / 255.0),
            label=label,
            filename=image_path.name,
        )
