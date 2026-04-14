from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ocr_project.common.types import DatasetItem
from ocr_project.stage1_data.collection_spec import CollectionSpec, default_collection_spec


def split_dataset(
    items: Iterable[DatasetItem],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[DatasetItem]]:
    """OCR-D04: Split dataset into train/val/test buckets."""
    items = list(items)
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def validate_dataset_quality(items: Iterable[DatasetItem]) -> list[str]:
    """Return human-readable dataset issues for inspection."""
    issues: list[str] = []
    for item in items:
        if not item.text and not item.boxes:
            issues.append(f"missing label: {item.image_path}")
    return issues


def check_required_paths(root_dir: Path) -> list[str]:
    """Validate that OCR-D01 directory layout is present."""
    spec = default_collection_spec()
    required = [
        root_dir / "data" / "masked",
        root_dir / "data" / "rejected",
        root_dir / "data" / "meta" / "collection_log.csv",
        root_dir / "data" / "meta" / "quality_check.csv",
    ]
    issues: list[str] = []
    for path in required:
        if not path.exists():
            issues.append(f"missing required path: {path}")
    if spec.total_min_samples <= 0:
        issues.append("invalid total_min_samples")
    return issues


def evaluate_definition_of_done(
    passed_items: Iterable[DatasetItem],
    spec: CollectionSpec | None = None,
) -> list[str]:
    """Return unmet OCR-D01 completion requirements."""
    spec = spec or default_collection_spec()
    items = list(passed_items)
    issues: list[str] = []

    if len(items) < spec.total_min_samples:
        issues.append(
            f"insufficient passed images: {len(items)} < {spec.total_min_samples}"
        )

    for format_code, required_count in spec.format_targets.items():
        actual_count = sum(
            1 for item in items if item.metadata.get("format") == format_code
        )
        if actual_count < required_count:
            issues.append(
                f"format quota not met for {format_code}: {actual_count} < {required_count}"
            )

    for industry_code, required_count in spec.industry_minimums.items():
        actual_count = sum(
            1 for item in items if item.metadata.get("industry") == industry_code
        )
        if actual_count < required_count:
            issues.append(
                f"industry quota not met for {industry_code}: {actual_count} < {required_count}"
            )

    unmasked = [item.image_path.name for item in items if item.metadata.get("masked") != "Y"]
    if unmasked:
        issues.append(f"unmasked items remain: {len(unmasked)}")

    return issues

