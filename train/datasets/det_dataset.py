from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr_project.stage2_preprocess.augmentation import AugmentPipeline
from ocr_project.stage2_preprocess.preprocess import PreprocessPipeline

from train.datasets.common import hwc_to_chw, load_image


@dataclass(slots=True)
class DetSample:
    image: np.ndarray
    prob_map: np.ndarray
    threshold_map: np.ndarray
    valid_mask: np.ndarray
    bboxes: list[list[int]]
    filename: str


class DetDataset:
    def __init__(
        self,
        data_dir: str | Path,
        label_file: str | Path,
        preprocess_config: str | Path,
        augment_config: str | Path,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.label_file = Path(label_file)
        self.preprocess = PreprocessPipeline(preprocess_config)
        self.augment = AugmentPipeline(augment_config)
        self.records = parse_det_gt(self.label_file)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DetSample:
        record = self.records[index]
        image = load_image(self.data_dir / record["filename"])
        processed = self.preprocess.run(image, mode="det")
        augmented = self.augment.run_det((processed["image"] * 255.0).astype(np.uint8), record["bboxes"])
        image_tensor = hwc_to_chw(augmented["image"] / 255.0)
        prob_map, threshold_map, valid_mask = make_det_targets(augmented["image"].shape[:2], augmented["bboxes"])
        return DetSample(
            image=image_tensor,
            prob_map=prob_map[None, ...],
            threshold_map=threshold_map[None, ...],
            valid_mask=valid_mask[None, ...],
            bboxes=augmented["bboxes"],
            filename=record["filename"],
        )


def parse_det_gt(label_file: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in Path(label_file).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or "\t" not in raw_line:
            continue
        filename, payload = raw_line.split("\t", 1)
        polygons = json.loads(payload)
        bboxes = []
        for polygon in polygons:
            points = polygon["points"]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            bboxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
        records.append({"filename": filename, "bboxes": bboxes})
    return records


def make_det_targets(image_shape: tuple[int, int], bboxes: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape
    out_h = max(1, height // 4)
    out_w = max(1, width // 4)
    prob_map = np.zeros((out_h, out_w), dtype=np.float32)
    threshold_map = np.zeros((out_h, out_w), dtype=np.float32)
    valid_mask = np.ones((out_h, out_w), dtype=np.float32)
    for bbox in bboxes:
        x1, y1, x2, y2 = [max(0, value // 4) for value in bbox]
        x2 = min(out_w - 1, x2)
        y2 = min(out_h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        prob_map[y1:y2, x1:x2] = 1.0
        threshold_map[y1:y2, x1:x2] = 1.0
    prob_map = cv2.GaussianBlur(prob_map, (0, 0), sigmaX=1.5)
    return prob_map, threshold_map, valid_mask
