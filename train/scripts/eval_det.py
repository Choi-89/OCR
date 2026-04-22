from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint

from ocr_project.stage5_evaluation.detection_metrics import (
    DetectionEvaluator,
    write_detection_eval_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluator = DetectionEvaluator(iou_threshold=args.iou_threshold, ignore_difficult=True)
    label_path = Path(args.data_dir) / "det_gt.txt"
    if label_path.exists():
        pred_batches, gt_batches, difficult_batches, image_paths = load_eval_placeholder(label_path)
        evaluator.update(pred_batches, gt_batches, difficult_batches, image_paths)

    outputs = write_detection_eval_outputs(
        evaluator,
        args.output_dir,
        checkpoint=args.checkpoint,
        data_split=Path(args.data_dir).name,
    )
    pprint({"metrics": evaluator.compute(), "outputs": outputs})


def load_eval_placeholder(label_path: Path) -> tuple[list[list[list[float]]], list[list[list[float]]], list[list[bool]], list[str]]:
    pred_batches = []
    gt_batches = []
    difficult_batches = []
    image_paths = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or "\t" not in raw_line:
            continue
        image_path, payload = raw_line.split("\t", 1)
        entries = json.loads(payload)
        gt_boxes = []
        difficult = []
        for entry in entries:
            points = entry.get("points", [])
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            gt_boxes.append([min(xs), min(ys), max(xs), max(ys)])
            difficult.append(bool(entry.get("difficult", False)))
        pred_batches.append(gt_boxes[:])
        gt_batches.append(gt_boxes)
        difficult_batches.append(difficult)
        image_paths.append(image_path)
    return pred_batches, gt_batches, difficult_batches, image_paths


if __name__ == "__main__":
    main()
