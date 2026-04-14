from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ocr_project.stage1_data.annotation_guide import AnnotationPolicy, default_annotation_policy


ANNOTATION_LOG_COLUMNS: tuple[str, ...] = (
    "filename",
    "annotator",
    "date",
    "bbox_count",
    "difficult_count",
    "unreadable_count",
    "reviewer",
    "review_date",
    "status",
)


@dataclass(slots=True)
class OCRPolygon:
    transcription: str
    points: list[list[int]]
    difficult: bool = False


@dataclass(slots=True)
class AnnotationRecord:
    filename: str
    annotator: str
    date: str
    bbox_count: int
    difficult_count: int
    unreadable_count: int
    reviewer: str
    review_date: str
    status: str


def initialize_annotation_workspace(root_dir: Path) -> dict[str, Path]:
    """Create OCR-D03 label output folders and log files."""
    policy = default_annotation_policy()
    required_dirs = [
        root_dir / "data" / "labels",
        root_dir / "data" / "labels" / "crop",
        root_dir / "data" / "labels" / "difficult",
        root_dir / "data" / "labels" / "unreadable",
        root_dir / "data" / "meta",
        root_dir / "data" / "meta" / "calibration",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    det_gt = root_dir / policy.detection_output
    rec_gt = root_dir / policy.recognition_output
    annotation_log = root_dir / policy.annotation_log
    calibration_manifest = root_dir / policy.calibration_dir / "manifest.json"

    det_gt.touch(exist_ok=True)
    rec_gt.touch(exist_ok=True)
    _ensure_csv_header(annotation_log, ANNOTATION_LOG_COLUMNS)
    if not calibration_manifest.exists():
        calibration_manifest.write_text(
            json.dumps(
                {
                    "target_image_count": policy.calibration_image_count,
                    "min_mean_iou": policy.min_mean_iou,
                    "cross_review_ratio": policy.cross_review_ratio,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "labels_root": root_dir / "data" / "labels",
        "det_gt": det_gt,
        "rec_gt": rec_gt,
        "crop_dir": root_dir / policy.crop_dir,
        "difficult_dir": root_dir / policy.difficult_dir,
        "unreadable_dir": root_dir / policy.unreadable_dir,
        "annotation_log": annotation_log,
        "calibration_dir": root_dir / policy.calibration_dir,
    }


def recommended_ppocrlabel_commands() -> list[str]:
    """Document the PPOCRLabel install and run commands."""
    return [
        "pip install PPOCRLabel",
        "PPOCRLabel --lang ch",
    ]


def ppocrlabel_shortcuts() -> dict[str, str]:
    """Provide the key shortcuts that annotators need."""
    return {
        "W": "draw_new_bbox",
        "D": "next_image",
        "A": "previous_image",
        "Ctrl+S": "save",
        "Q": "run_auto_recognition",
    }


def format_det_gt_line(image_path: str, polygons: list[OCRPolygon]) -> str:
    """Serialize one PaddleOCR detection ground-truth line."""
    payload = [
        {
            "transcription": polygon.transcription,
            "points": polygon.points,
            "difficult": polygon.difficult,
        }
        for polygon in polygons
    ]
    return f"{image_path}\t{json.dumps(payload, ensure_ascii=False)}"


def format_rec_gt_line(crop_path: str, text: str) -> str:
    """Serialize one PaddleOCR recognition ground-truth line."""
    return f"{crop_path} {text}"


def append_det_gt(det_gt_path: Path, image_path: str, polygons: list[OCRPolygon]) -> None:
    with det_gt_path.open("a", encoding="utf-8") as handle:
        handle.write(format_det_gt_line(image_path, polygons))
        handle.write("\n")


def append_rec_gt(rec_gt_path: Path, crop_path: str, text: str) -> None:
    with rec_gt_path.open("a", encoding="utf-8") as handle:
        handle.write(format_rec_gt_line(crop_path, text))
        handle.write("\n")


def append_annotation_log(csv_path: Path, record: AnnotationRecord) -> None:
    _append_csv_row(
        csv_path,
        {
            "filename": record.filename,
            "annotator": record.annotator,
            "date": record.date,
            "bbox_count": record.bbox_count,
            "difficult_count": record.difficult_count,
            "unreadable_count": record.unreadable_count,
            "reviewer": record.reviewer,
            "review_date": record.review_date,
            "status": record.status,
        },
    )


def polygon_mode_required(tilt_degrees: float, policy: AnnotationPolicy | None = None) -> bool:
    policy = policy or default_annotation_policy()
    return abs(tilt_degrees) > policy.polygon_angle_threshold_degrees


def difficult_or_unreadable(text: str, difficult: bool, policy: AnnotationPolicy | None = None) -> str:
    policy = policy or default_annotation_policy()
    if text == policy.unreadable_token:
        return "unreadable"
    if difficult:
        return "difficult"
    return "normal"


def calculate_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute IoU for axis-aligned boxes in [x1, y1, x2, y2]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denominator = area_a + area_b - intersection
    if denominator <= 0:
        return 0.0
    return intersection / denominator


def mean_iou(paired_boxes: list[tuple[list[int], list[int]]]) -> float:
    """Average IoU across calibration annotations."""
    if not paired_boxes:
        return 0.0
    return mean(calculate_iou(first, second) for first, second in paired_boxes)


def passes_iou_calibration(paired_boxes: list[tuple[list[int], list[int]]]) -> bool:
    """Check whether the annotator calibration passes the 0.85 IoU threshold."""
    policy = default_annotation_policy()
    return mean_iou(paired_boxes) >= policy.min_mean_iou


def evaluate_annotation_definition_of_done(
    masked_image_count: int,
    det_gt_exists: bool,
    rec_gt_exists: bool,
    crop_count: int,
    unreadable_box_count: int,
    total_box_count: int,
    mean_calibration_iou: float,
    cross_reviewed_image_count: int,
    annotation_log_exists: bool,
) -> list[str]:
    """Return unmet OCR-D03 completion requirements."""
    policy = default_annotation_policy()
    issues: list[str] = []

    if not det_gt_exists:
        issues.append("det_gt_missing")
    if not rec_gt_exists:
        issues.append("rec_gt_missing")
    if crop_count <= 0:
        issues.append("crop_images_missing")
    if total_box_count > 0 and unreadable_box_count / total_box_count > policy.max_unreadable_ratio:
        issues.append("unreadable_ratio_exceeds_5_percent")
    if mean_calibration_iou < policy.min_mean_iou:
        issues.append(
            f"mean_calibration_iou_below_threshold: {mean_calibration_iou:.3f} < {policy.min_mean_iou:.2f}"
        )
    minimum_cross_reviews = int(masked_image_count * policy.cross_review_ratio)
    if cross_reviewed_image_count < minimum_cross_reviews:
        issues.append(
            f"cross_review_count_below_threshold: {cross_reviewed_image_count} < {minimum_cross_reviews}"
        )
    if not annotation_log_exists:
        issues.append("annotation_log_missing")

    return issues


def calibration_manifest() -> dict[str, Any]:
    """Expose the calibration policy for downstream tooling."""
    policy = default_annotation_policy()
    return {
        "calibration_image_count": policy.calibration_image_count,
        "min_mean_iou": policy.min_mean_iou,
        "cross_review_ratio": policy.cross_review_ratio,
        "review_batch_size": policy.review_batch_size,
    }


def _ensure_csv_header(csv_path: Path, fieldnames: tuple[str, ...]) -> None:
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writerow(row)
