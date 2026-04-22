from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BBox = list[float]


@dataclass(slots=True)
class MatchResult:
    tp: int
    fp: int
    fn: int
    matches: list[tuple[int, int, float]]
    fp_indices: list[int]
    fn_indices: list[int]
    difficult_matched: int = 0


@dataclass(slots=True)
class ImageEvalRecord:
    image_path: str
    pred_boxes: list[BBox]
    gt_boxes: list[BBox]
    difficult_flags: list[bool]
    match: MatchResult
    format_code: str = "unknown"
    industry: str = "unknown"


@dataclass(slots=True)
class DetectionEvaluator:
    iou_threshold: float = 0.5
    ignore_difficult: bool = True
    records: list[ImageEvalRecord] = field(default_factory=list)

    def update(
        self,
        pred_boxes: list[list[BBox]],
        gt_boxes: list[list[BBox]],
        difficult_flags: list[list[bool]] | None = None,
        image_paths: list[str] | None = None,
    ) -> None:
        difficult_flags = difficult_flags or [[False] * len(boxes) for boxes in gt_boxes]
        image_paths = image_paths or [f"image_{len(self.records) + index}" for index in range(len(pred_boxes))]
        for preds, gts, flags, image_path in zip(pred_boxes, gt_boxes, difficult_flags, image_paths):
            match = greedy_match(preds, gts, flags, self.iou_threshold, self.ignore_difficult)
            format_code, industry = parse_filename_metadata(image_path)
            self.records.append(
                ImageEvalRecord(
                    image_path=image_path,
                    pred_boxes=preds,
                    gt_boxes=gts,
                    difficult_flags=flags,
                    match=match,
                    format_code=format_code,
                    industry=industry,
                )
            )

    def compute(self) -> dict[str, Any]:
        tp = sum(record.match.tp for record in self.records)
        fp = sum(record.match.fp for record in self.records)
        fn = sum(record.match.fn for record in self.records)
        return build_metric_dict(tp, fp, fn, self.iou_threshold)

    def compute_per_image(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in self.records:
            metrics = build_metric_dict(record.match.tp, record.match.fp, record.match.fn, self.iou_threshold)
            rows.append(
                {
                    "image_path": record.image_path,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "num_pred": len(record.pred_boxes),
                    "num_gt": sum(1 for flag in record.difficult_flags if not (flag and self.ignore_difficult)),
                    "format": record.format_code,
                    "industry": record.industry,
                }
            )
        return rows

    def breakdown(self, key: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[ImageEvalRecord]] = defaultdict(list)
        for record in self.records:
            grouped[getattr(record, key)].append(record)
        rows: list[dict[str, Any]] = []
        for group_name, records in sorted(grouped.items()):
            tp = sum(record.match.tp for record in records)
            fp = sum(record.match.fp for record in records)
            fn = sum(record.match.fn for record in records)
            metrics = build_metric_dict(tp, fp, fn, self.iou_threshold)
            rows.append({"group": group_name, "sample_count": len(records), **metrics})
        return rows

    def reset(self) -> None:
        self.records.clear()

    def visualize(
        self,
        image: np.ndarray,
        pred_boxes: list[BBox],
        gt_boxes: list[BBox],
        save_path: str | Path,
        difficult_flags: list[bool] | None = None,
    ) -> Path:
        flags = difficult_flags or [False] * len(gt_boxes)
        match = greedy_match(pred_boxes, gt_boxes, flags, self.iou_threshold, self.ignore_difficult)
        return visualize_detection_result(image, pred_boxes, gt_boxes, match, save_path)


def detection_metric_names() -> list[str]:
    return ["precision", "recall", "f1", "iou", "tp", "fp", "fn"]


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = normalize_box(box_a)
    bx1, by1, bx2, by2 = normalize_box(box_b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def normalize_box(box: BBox) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def greedy_match(
    pred_boxes: list[BBox],
    gt_boxes: list[BBox],
    difficult_flags: list[bool] | None = None,
    iou_threshold: float = 0.5,
    ignore_difficult: bool = True,
) -> MatchResult:
    difficult_flags = difficult_flags or [False] * len(gt_boxes)
    pairs: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(pred_boxes):
        for gt_index, gt in enumerate(gt_boxes):
            iou = bbox_iou(pred, gt)
            if iou >= iou_threshold:
                pairs.append((iou, pred_index, gt_index))
    pairs.sort(reverse=True, key=lambda item: item[0])

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    difficult_matched = 0
    for iou, pred_index, gt_index in pairs:
        if pred_index in matched_preds or gt_index in matched_gts:
            continue
        matched_preds.add(pred_index)
        matched_gts.add(gt_index)
        matches.append((pred_index, gt_index, iou))
        if difficult_flags[gt_index]:
            difficult_matched += 1

    fp_indices = [index for index in range(len(pred_boxes)) if index not in matched_preds]
    fn_indices = [
        index
        for index in range(len(gt_boxes))
        if index not in matched_gts and not (ignore_difficult and difficult_flags[index])
    ]
    return MatchResult(
        tp=len(matches),
        fp=len(fp_indices),
        fn=len(fn_indices),
        matches=matches,
        fp_indices=fp_indices,
        fn_indices=fn_indices,
        difficult_matched=difficult_matched,
    )


def build_metric_dict(tp: int, fp: int, fn: int, iou_threshold: float) -> dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_pred": tp + fp,
        "num_gt": tp + fn,
        "iou_threshold": iou_threshold,
    }


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def parse_filename_metadata(image_path: str) -> tuple[str, str]:
    stem = Path(image_path).stem
    parts = stem.split("_")
    if stem.startswith("synth_") and len(parts) >= 4:
        return parts[1], parts[2]
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "unknown", "unknown"


def analyze_errors(records: list[ImageEvalRecord], min_box_size: int = 3) -> dict[str, Counter[str]]:
    fp_counter: Counter[str] = Counter()
    fn_counter: Counter[str] = Counter()
    gt_areas = [box_area(gt) for record in records for gt in record.gt_boxes]
    avg_gt_area = sum(gt_areas) / len(gt_areas) if gt_areas else 0.0

    for record in records:
        gt_match_counts = Counter(gt_idx for _, gt_idx, _ in record.match.matches)
        for pred_index in record.match.fp_indices:
            pred = record.pred_boxes[pred_index]
            if is_extreme_aspect_ratio(pred):
                fp_counter["grid_line_false_positive"] += 1
            elif has_duplicate_overlap(pred, record.gt_boxes, gt_match_counts):
                fp_counter["duplicate_detection"] += 1
            elif box_area(pred) < (min_box_size * min_box_size * 2):
                fp_counter["small_noise"] += 1
            else:
                fp_counter["other_false_positive"] += 1

        for gt_index in record.match.fn_indices:
            gt = record.gt_boxes[gt_index]
            if avg_gt_area and box_area(gt) < avg_gt_area * 0.3:
                fn_counter["small_cell_miss"] += 1
            elif is_dense_box(gt, record.gt_boxes):
                fn_counter["dense_region_miss"] += 1
            elif gt_index < len(record.difficult_flags) and not record.difficult_flags[gt_index]:
                fn_counter["blur_or_hard_text_miss"] += 1
            else:
                fn_counter["other_false_negative"] += 1
    return {"fp": fp_counter, "fn": fn_counter}


def box_area(box: BBox) -> float:
    x1, y1, x2, y2 = normalize_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def is_extreme_aspect_ratio(box: BBox) -> bool:
    x1, y1, x2, y2 = normalize_box(box)
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    ratio = width / height
    return ratio > 10 or ratio < 0.1


def has_duplicate_overlap(pred: BBox, gt_boxes: list[BBox], gt_match_counts: Counter[int]) -> bool:
    return any(bbox_iou(pred, gt) > 0.3 and gt_match_counts.get(index, 0) >= 1 for index, gt in enumerate(gt_boxes))


def is_dense_box(target: BBox, boxes: list[BBox], distance_threshold: float = 5.0) -> bool:
    tx1, ty1, tx2, ty2 = normalize_box(target)
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2
    for box in boxes:
        if box == target:
            continue
        x1, y1, x2, y2 = normalize_box(box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if abs(cx - tcx) < distance_threshold or abs(cy - tcy) < distance_threshold:
            return True
    return False


def write_detection_eval_outputs(
    evaluator: DetectionEvaluator,
    output_dir: str | Path,
    *,
    checkpoint: str,
    data_split: str,
    metrics_iou_03: dict[str, Any] | None = None,
    target_f1: float = 0.88,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluator.compute()
    summary = {
        "eval_date": str(date.today()),
        "checkpoint": checkpoint,
        "data_split": data_split,
        "iou_threshold": evaluator.iou_threshold,
        "metrics": metrics,
        "metrics_iou_03": metrics_iou_03 or {},
        "target_met": metrics["f1"] >= target_f1,
        "notes": "",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    per_image_path = output / "per_image.csv"
    write_csv(per_image_path, evaluator.compute_per_image())
    format_path = output / "format_breakdown.csv"
    write_csv(format_path, evaluator.breakdown("format_code"))
    industry_path = output / "industry_breakdown.csv"
    write_csv(industry_path, evaluator.breakdown("industry"))
    error_path = output / "error_analysis.md"
    error_path.write_text(build_error_report(evaluator), encoding="utf-8")
    for subdir in ("visualizations/tp_fp_fn_samples", "visualizations/low_f1_samples", "visualizations/probability_maps"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    return {
        "summary": str(summary_path),
        "per_image": str(per_image_path),
        "format_breakdown": str(format_path),
        "industry_breakdown": str(industry_path),
        "error_analysis": str(error_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_error_report(evaluator: DetectionEvaluator) -> str:
    analysis = analyze_errors(evaluator.records)
    fp_total = sum(analysis["fp"].values())
    fn_total = sum(analysis["fn"].values())
    low_f1 = sorted(evaluator.compute_per_image(), key=lambda row: row["f1"])[:5]
    lines = [
        "# Detection Error Analysis Report",
        "",
        "## Summary",
        f"- total FP: {fp_total}",
        f"- total FN: {fn_total}",
        "",
        "## FP Type Distribution",
        "| type | count | ratio |",
        "|---|---:|---:|",
    ]
    for label, count in analysis["fp"].items():
        lines.append(f"| {label} | {count} | {safe_div(count, fp_total):.1%} |")
    lines.extend(["", "## FN Type Distribution", "| type | count | ratio |", "|---|---:|---:|"])
    for label, count in analysis["fn"].items():
        lines.append(f"| {label} | {count} | {safe_div(count, fn_total):.1%} |")
    lines.extend(["", "## Lowest F1 Images Top 5"])
    for index, row in enumerate(low_f1, start=1):
        lines.append(f"{index}. {row['image_path']} (F1: {row['f1']:.3f})")
    return "\n".join(lines) + "\n"


def visualize_detection_result(
    image: np.ndarray,
    pred_boxes: list[BBox],
    gt_boxes: list[BBox],
    match: MatchResult,
    save_path: str | Path,
) -> Path:
    canvas = image.copy()
    matched_preds = {pred_idx for pred_idx, _, _ in match.matches}
    matched_gts = {gt_idx for _, gt_idx, _ in match.matches}
    for gt_index, gt in enumerate(gt_boxes):
        color = (0, 255, 0) if gt_index in matched_gts else (255, 0, 0)
        draw_box(canvas, gt, color)
    for pred_index, pred in enumerate(pred_boxes):
        if pred_index not in matched_preds:
            draw_box(canvas, pred, (0, 0, 255))
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), canvas)
    return target


def draw_box(image: np.ndarray, box: BBox, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in normalize_box(box)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


def evaluate_detection(
    pred_batches: list[list[BBox]],
    gt_batches: list[list[BBox]],
    difficult_batches: list[list[bool]] | None = None,
    image_paths: list[str] | None = None,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    evaluator = DetectionEvaluator(iou_threshold=iou_threshold)
    evaluator.update(pred_batches, gt_batches, difficult_batches, image_paths)
    return evaluator.compute()
