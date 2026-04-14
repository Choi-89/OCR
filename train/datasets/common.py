from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(slots=True)
class SampleRecord:
    image_path: Path
    label: str


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed_to_open_image:{path}")
    return image


def parse_rec_gt(label_file: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for raw_line in Path(label_file).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or " " not in raw_line:
            continue
        image_path, text = raw_line.split(" ", 1)
        records.append(SampleRecord(image_path=Path(image_path), label=text))
    return records


def pad_width_to_batch(images: Iterable[np.ndarray], padding_value: float = 0.0) -> np.ndarray:
    items = list(images)
    if not items:
        raise ValueError("empty_batch")
    max_h = max(item.shape[1] for item in items)
    max_w = max(item.shape[2] for item in items)
    batch = np.full((len(items), items[0].shape[0], max_h, max_w), padding_value, dtype=np.float32)
    for index, item in enumerate(items):
        _, height, width = item.shape
        batch[index, :, :height, :width] = item
    return batch


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    return np.transpose(image, (2, 0, 1)).astype(np.float32)
