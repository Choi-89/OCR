from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from ocr_project.stage1_data.collection_spec import (
    ALLOWED_EXTENSIONS,
    ALLOWED_FORMATS,
    ALLOWED_INDUSTRIES,
    CollectionSpec,
    QualityThreshold,
    default_collection_spec,
    expected_data_directories,
)


FILENAME_PATTERN = re.compile(
    r"^(paper|scan|screen|excel|handwrite)_"
    r"(hospital|convenience|factory|office|food|etc)_"
    r"(?P<seq>\d{4})\.(jpg|jpeg|png|pdf)$",
    re.IGNORECASE,
)

COLLECTION_LOG_COLUMNS: tuple[str, ...] = (
    "filename",
    "collected_date",
    "format",
    "industry",
    "resolution",
    "status",
    "reject_reason",
    "masked",
)

QUALITY_CHECK_COLUMNS: tuple[str, ...] = (
    "filename",
    "passed",
    "reject_reason",
    "short_side_px",
    "text_area_ratio",
    "tilt_degrees",
    "legibility_ratio",
    "file_size_kb",
    "contains_personal_info",
    "duplicate_hash",
)


@dataclass(slots=True)
class ImageQualitySnapshot:
    width: int
    height: int
    text_area_ratio: float
    tilt_degrees: float
    legibility_ratio: float
    file_size_bytes: int
    is_duplicate_hash: bool = False
    is_non_schedule_image: bool = False
    has_motion_blur: bool = False
    masked_text_ratio: float = 0.0

    @property
    def short_side_px(self) -> int:
        return min(self.width, self.height)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def file_size_kb(self) -> float:
        return round(self.file_size_bytes / 1024, 2)


@dataclass(slots=True)
class CollectionRecord:
    filename: str
    collected_date: str
    format: str
    industry: str
    resolution: str
    status: str
    reject_reason: str
    masked: str


def initialize_collection_workspace(
    root_dir: Path,
    spec: CollectionSpec | None = None,
) -> dict[str, Path]:
    """Create OCR-D01 folders and metadata CSV files."""
    spec = spec or default_collection_spec()
    del spec  # Reserved for future per-project overrides.

    for directory in expected_data_directories(root_dir):
        directory.mkdir(parents=True, exist_ok=True)

    collection_log = root_dir / "data" / "meta" / "collection_log.csv"
    quality_check = root_dir / "data" / "meta" / "quality_check.csv"
    _ensure_csv_header(collection_log, COLLECTION_LOG_COLUMNS)
    _ensure_csv_header(quality_check, QUALITY_CHECK_COLUMNS)

    return {
        "collection_log": collection_log,
        "quality_check": quality_check,
        "masked_dir": root_dir / "data" / "masked",
        "rejected_dir": root_dir / "data" / "rejected",
    }


def build_filename(
    format_code: str,
    industry_code: str,
    sequence: int,
    extension: str,
) -> str:
    """Generate an OCR-D01 compliant filename."""
    normalized_extension = extension.lower()
    if not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"

    if format_code not in ALLOWED_FORMATS:
        raise ValueError(f"unsupported format code: {format_code}")
    if industry_code not in ALLOWED_INDUSTRIES:
        raise ValueError(f"unsupported industry code: {industry_code}")
    if normalized_extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported extension: {normalized_extension}")

    return f"{format_code}_{industry_code}_{sequence:04d}{normalized_extension}"


def validate_filename(filename: str) -> bool:
    """Check whether the filename follows the OCR-D01 naming rule."""
    return bool(FILENAME_PATTERN.match(filename))


def assess_image_quality(
    snapshot: ImageQualitySnapshot,
    thresholds: QualityThreshold | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate pass/fail conditions from OCR-D01."""
    thresholds = thresholds or default_collection_spec().quality
    reasons: list[str] = []

    if snapshot.short_side_px < thresholds.min_short_side_px:
        reasons.append("resolution_below_720px")
    if snapshot.text_area_ratio < thresholds.min_text_area_ratio:
        reasons.append("text_area_below_20_percent")
    if snapshot.has_motion_blur:
        reasons.append("motion_blur_or_focus_failure")
    if snapshot.masked_text_ratio > 0.5:
        reasons.append("masking_covers_over_50_percent_of_text")
    if snapshot.is_non_schedule_image:
        reasons.append("non_schedule_image")
    if snapshot.is_duplicate_hash:
        reasons.append("duplicate_image_hash")
    if abs(snapshot.tilt_degrees) > thresholds.max_tilt_degrees:
        reasons.append("tilt_exceeds_30_degrees")
    if snapshot.legibility_ratio < thresholds.min_legibility_ratio:
        reasons.append("legibility_below_80_percent")
    if snapshot.file_size_kb < thresholds.min_file_size_kb:
        reasons.append("file_size_below_100kb")
    if snapshot.file_size_bytes > thresholds.max_file_size_mb * 1024 * 1024:
        reasons.append("file_size_above_20mb")

    return not reasons, reasons


def append_collection_log(csv_path: Path, record: CollectionRecord) -> None:
    """Append one row to collection_log.csv."""
    _append_csv_row(
        csv_path,
        {
            "filename": record.filename,
            "collected_date": record.collected_date,
            "format": record.format,
            "industry": record.industry,
            "resolution": record.resolution,
            "status": record.status,
            "reject_reason": record.reject_reason,
            "masked": record.masked,
        },
    )


def append_quality_check(
    csv_path: Path,
    filename: str,
    snapshot: ImageQualitySnapshot,
    passed: bool,
    reject_reasons: list[str],
    contains_personal_info: bool,
) -> None:
    """Append one row to quality_check.csv."""
    _append_csv_row(
        csv_path,
        {
            "filename": filename,
            "passed": "Y" if passed else "N",
            "reject_reason": "|".join(reject_reasons),
            "short_side_px": snapshot.short_side_px,
            "text_area_ratio": snapshot.text_area_ratio,
            "tilt_degrees": snapshot.tilt_degrees,
            "legibility_ratio": snapshot.legibility_ratio,
            "file_size_kb": snapshot.file_size_kb,
            "contains_personal_info": "Y" if contains_personal_info else "N",
            "duplicate_hash": "Y" if snapshot.is_duplicate_hash else "N",
        },
    )


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
