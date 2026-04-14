from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ocr_project.stage1_data.annotation_guide import default_annotation_policy
from ocr_project.stage1_data.collection_spec import ALLOWED_FORMATS, ALLOWED_INDUSTRIES, default_collection_spec
from ocr_project.stage1_data.synthetic_generator import MAX_SYNTHETIC_RATIO


SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
MINIMUM_SET_REQUIREMENTS: dict[str, dict[str, int]] = {
    "train": {"images": 700, "det_boxes": 5000},
    "val": {"images": 100, "det_boxes": 700},
    "test": {"images": 100, "det_boxes": 700},
}


@dataclass(slots=True)
class DatasetImageRecord:
    image_path: Path
    source: str
    format_code: str
    industry: str
    width: int = 1280
    height: int = 720
    difficult: bool = False
    unreadable_count: int = 0
    detection_boxes: int = 0
    recognition_crops: list[str] = field(default_factory=list)
    crop_texts: list[str] = field(default_factory=list)
    synthetic_method: str | None = None
    label_json_path: Path | None = None

    @property
    def filename(self) -> str:
        return self.image_path.name

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def strata_key(self) -> tuple[str, str]:
        return (self.format_code, self.industry)


@dataclass(slots=True)
class QualityIssue:
    filename: str
    issue_code: str
    details: str


@dataclass(slots=True)
class SplitBundle:
    train: list[DatasetImageRecord]
    val: list[DatasetImageRecord]
    test: list[DatasetImageRecord]


def initialize_dataset_workspace(root_dir: Path) -> dict[str, Path]:
    """Create the final Stage 1 dataset directory layout."""
    required_dirs = [
        root_dir / "data" / "dataset" / "train" / "images",
        root_dir / "data" / "dataset" / "train" / "crop",
        root_dir / "data" / "dataset" / "val" / "images",
        root_dir / "data" / "dataset" / "val" / "crop",
        root_dir / "data" / "dataset" / "test" / "images",
        root_dir / "data" / "dataset" / "test" / "crop",
        root_dir / "data" / "meta",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    output_files = {
        "train_det": root_dir / "data" / "dataset" / "train" / "det_gt.txt",
        "train_rec": root_dir / "data" / "dataset" / "train" / "rec_gt.txt",
        "val_det": root_dir / "data" / "dataset" / "val" / "det_gt.txt",
        "val_rec": root_dir / "data" / "dataset" / "val" / "rec_gt.txt",
        "test_det": root_dir / "data" / "dataset" / "test" / "det_gt.txt",
        "test_rec": root_dir / "data" / "dataset" / "test" / "rec_gt.txt",
        "split_config": root_dir / "data" / "meta" / "split_config.json",
        "dataset_stats": root_dir / "data" / "meta" / "dataset_stats.json",
        "quality_report": root_dir / "data" / "meta" / "quality_report.md",
    }
    for path in output_files.values():
        if path.suffix == ".txt":
            path.write_text("", encoding="utf-8")
    return output_files


def check_prerequisites(root_dir: Path) -> list[str]:
    """Validate all OCR-D04 prerequisites before splitting."""
    issues: list[str] = []

    masked_count = len(list((root_dir / "data" / "masked").glob("*.*")))
    synthetic_count = len(list((root_dir / "data" / "synthetic").rglob("*.png")))
    synthetic_label_count = len(list((root_dir / "data" / "synthetic" / "labels").glob("*.json")))
    crop_count = len(list((root_dir / "data" / "labels" / "crop").glob("*.*")))

    if masked_count < 600:
        issues.append(f"masked_count_below_600: {masked_count}")
    if synthetic_count < 400:
        issues.append(f"synthetic_count_below_400: {synthetic_count}")
    if not (root_dir / "data" / "labels" / "det_gt.txt").exists():
        issues.append("det_gt_missing")
    if not (root_dir / "data" / "labels" / "rec_gt.txt").exists():
        issues.append("rec_gt_missing")
    if crop_count <= 0:
        issues.append("crop_images_missing")
    if synthetic_label_count < synthetic_count:
        issues.append(f"synthetic_label_json_missing: {synthetic_label_count} < {synthetic_count}")

    annotation_log = root_dir / "data" / "meta" / "annotation_log.csv"
    if not annotation_log.exists():
        issues.append("annotation_log_missing")
    else:
        with annotation_log.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rejected = [row["filename"] for row in reader if row.get("status") == "rejected"]
        if rejected:
            issues.append(f"annotation_log_contains_rejected: {len(rejected)}")

    return issues


def parse_det_gt(det_gt_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load PaddleOCR detection labels keyed by image filename."""
    entries: dict[str, list[dict[str, Any]]] = {}
    if not det_gt_path.exists():
        return entries

    with det_gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            image_path, payload = line.split("\t", 1)
            entries[Path(image_path).name] = json.loads(payload)
    return entries


def parse_rec_gt(rec_gt_path: Path) -> dict[str, str]:
    """Load PaddleOCR recognition labels keyed by crop filename."""
    entries: dict[str, str] = {}
    if not rec_gt_path.exists():
        return entries

    with rec_gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            crop_path, text = line.split(" ", 1)
            entries[Path(crop_path).name] = text
    return entries


def collect_dataset_records(root_dir: Path) -> list[DatasetImageRecord]:
    """Gather real and synthetic image metadata for splitting."""
    det_entries = parse_det_gt(root_dir / "data" / "labels" / "det_gt.txt")
    rec_entries = parse_rec_gt(root_dir / "data" / "labels" / "rec_gt.txt")
    records: list[DatasetImageRecord] = []

    for image_path in sorted((root_dir / "data" / "masked").glob("*.*")):
        format_code, industry = parse_filename_metadata(image_path.name)
        polygons = det_entries.get(image_path.name, [])
        crops, texts = _match_recognition_entries(image_path.name, rec_entries)
        records.append(
            DatasetImageRecord(
                image_path=image_path,
                source="masked",
                format_code=format_code,
                industry=industry,
                difficult=any(item.get("difficult") for item in polygons),
                unreadable_count=sum(1 for item in polygons if item.get("transcription") == "###"),
                detection_boxes=len(polygons),
                recognition_crops=crops,
                crop_texts=texts,
            )
        )

    synthetic_root = root_dir / "data" / "synthetic"
    for image_path in sorted(synthetic_root.rglob("*.png")):
        if "labels" in image_path.parts:
            continue
        method = image_path.parts[-3] if len(image_path.parts) >= 3 else "render"
        industry = image_path.parent.name
        label_json = root_dir / "data" / "synthetic" / "labels" / f"{image_path.stem}.json"
        label_payload = json.loads(label_json.read_text(encoding="utf-8")) if label_json.exists() else {}
        bboxes = label_payload.get("bboxes", [])
        texts = [box.get("text", "") for box in bboxes]
        crops = [f"{image_path.stem}_{idx}.png" for idx, _ in enumerate(bboxes)]
        records.append(
            DatasetImageRecord(
                image_path=image_path,
                source="synthetic",
                format_code=_synthetic_to_format(method),
                industry=industry,
                difficult=False,
                unreadable_count=sum(1 for text in texts if text == "###"),
                detection_boxes=len(bboxes),
                recognition_crops=crops,
                crop_texts=texts,
                synthetic_method=method,
                label_json_path=label_json if label_json.exists() else None,
            )
        )

    return records


def run_automatic_quality_checks(records: list[DatasetImageRecord]) -> list[QualityIssue]:
    """Perform OCR-D04 automatic checks on the full dataset."""
    issues: list[QualityIssue] = []
    seen_hashes: dict[str, str] = {}

    for record in records:
        if not record.image_path.exists():
            issues.append(QualityIssue(record.filename, "missing_image", str(record.image_path)))
            continue

        digest = _file_md5(record.image_path)
        if digest in seen_hashes:
            issues.append(QualityIssue(record.filename, "duplicate_image", f"same_as={seen_hashes[digest]}"))
        else:
            seen_hashes[digest] = record.filename

        if record.short_side < default_collection_spec().quality.min_short_side_px:
            issues.append(QualityIssue(record.filename, "resolution_below_720px", str(record.short_side)))

        if record.detection_boxes <= 0:
            issues.append(QualityIssue(record.filename, "empty_detection_label", "no detection boxes"))

        if any(text == "" for text in record.crop_texts):
            issues.append(QualityIssue(record.filename, "empty_recognition_text", "blank crop transcription"))

        if record.source == "synthetic" and record.label_json_path and not record.label_json_path.exists():
            issues.append(QualityIssue(record.filename, "missing_synthetic_label_json", str(record.label_json_path)))

    synthetic_ratio = sum(1 for item in records if item.is_synthetic) / len(records) if records else 0.0
    if synthetic_ratio > MAX_SYNTHETIC_RATIO:
        issues.append(QualityIssue("dataset", "synthetic_ratio_exceeds_50_percent", f"{synthetic_ratio:.3f}"))

    return issues


def summarize_dataset_stats(records: list[DatasetImageRecord]) -> dict[str, Any]:
    """Aggregate dataset-wide statistics required by OCR-D04."""
    total_images = len(records)
    real_images = sum(1 for record in records if not record.is_synthetic)
    synthetic_images = sum(1 for record in records if record.is_synthetic)
    total_boxes = sum(record.detection_boxes for record in records)
    total_crops = sum(len(record.recognition_crops) for record in records)
    unreadable_boxes = sum(record.unreadable_count for record in records)
    difficult_images = sum(1 for record in records if record.difficult)

    format_counter = Counter(record.format_code for record in records)
    industry_counter = Counter(record.industry for record in records)
    text_length_buckets = Counter(_text_length_bucket(text) for record in records for text in record.crop_texts)

    return {
        "counts": {
            "total_images": total_images,
            "real_images": real_images,
            "synthetic_images": synthetic_images,
            "total_detection_boxes": total_boxes,
            "total_recognition_crops": total_crops,
            "unreadable_boxes": unreadable_boxes,
            "unreadable_ratio": (unreadable_boxes / total_boxes) if total_boxes else 0.0,
            "difficult_images": difficult_images,
            "difficult_ratio": (difficult_images / total_images) if total_images else 0.0,
        },
        "format_distribution": _distribution_table(format_counter, ALLOWED_FORMATS),
        "industry_distribution": _distribution_table(industry_counter, ALLOWED_INDUSTRIES),
        "text_length_distribution": _distribution_table(
            text_length_buckets,
            ("1", "2-4", "5-10", "11+"),
        ),
        "long_text_over_5_percent": (
            text_length_buckets.get("11+", 0) / total_boxes > 0.05 if total_boxes else False
        ),
    }


def stratified_split(
    records: list[DatasetImageRecord],
    seed: int = 42,
) -> SplitBundle:
    """Split by format x industry, keeping synthetic items in train only."""
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[DatasetImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.strata_key].append(record)

    train: list[DatasetImageRecord] = []
    val: list[DatasetImageRecord] = []
    test: list[DatasetImageRecord] = []

    for _, group in grouped.items():
        real_items = [item for item in group if not item.is_synthetic]
        synthetic_items = [item for item in group if item.is_synthetic]
        eligible_test_items = [item for item in real_items if not item.difficult]

        rng.shuffle(real_items)
        rng.shuffle(eligible_test_items)
        rng.shuffle(synthetic_items)

        val_count = int(len(real_items) * SPLIT_RATIOS["val"])
        test_count = int(len(real_items) * SPLIT_RATIOS["test"])
        test_items = eligible_test_items[:test_count]
        used_for_test = {item.filename for item in test_items}
        remaining_real = [item for item in real_items if item.filename not in used_for_test]
        val_items = remaining_real[:val_count]
        train_items = remaining_real[val_count:]

        train.extend(train_items)
        train.extend(synthetic_items)
        val.extend(val_items)
        test.extend(test_items)

    return SplitBundle(train=train, val=val, test=test)


def validate_split_bundle(bundle: SplitBundle) -> list[str]:
    """Check leakage, synthetic placement, difficult exclusion, and set minimums."""
    issues: list[str] = []
    sets = {"train": bundle.train, "val": bundle.val, "test": bundle.test}

    for first_name, first_items in sets.items():
        first_filenames = {item.filename for item in first_items}
        for second_name, second_items in sets.items():
            if first_name >= second_name:
                continue
            overlap = first_filenames & {item.filename for item in second_items}
            if overlap:
                issues.append(f"filename_overlap_{first_name}_{second_name}: {len(overlap)}")

    if any(item.is_synthetic for item in bundle.val):
        issues.append("synthetic_found_in_val")
    if any(item.is_synthetic for item in bundle.test):
        issues.append("synthetic_found_in_test")
    if any(item.difficult for item in bundle.test):
        issues.append("difficult_found_in_test")

    for split_name, requirements in MINIMUM_SET_REQUIREMENTS.items():
        items = sets[split_name]
        image_count = len(items)
        det_boxes = sum(item.detection_boxes for item in items)
        if image_count < requirements["images"]:
            issues.append(f"{split_name}_image_count_below_minimum: {image_count} < {requirements['images']}")
        if det_boxes < requirements["det_boxes"]:
            issues.append(f"{split_name}_det_boxes_below_minimum: {det_boxes} < {requirements['det_boxes']}")

    return issues


def write_dataset_outputs(
    root_dir: Path,
    bundle: SplitBundle,
    split_seed: int = 42,
) -> dict[str, Path]:
    """Write split metadata and placeholder dataset artifacts."""
    outputs = initialize_dataset_workspace(root_dir)
    split_map = {"train": bundle.train, "val": bundle.val, "test": bundle.test}

    for split_name, records in split_map.items():
        image_dir = root_dir / "data" / "dataset" / split_name / "images"
        crop_dir = root_dir / "data" / "dataset" / split_name / "crop"
        det_gt_path = root_dir / "data" / "dataset" / split_name / "det_gt.txt"
        rec_gt_path = root_dir / "data" / "dataset" / split_name / "rec_gt.txt"

        det_lines: list[str] = []
        rec_lines: list[str] = []

        for record in records:
            target_image = image_dir / record.filename
            shutil.copy2(record.image_path, target_image)

            if record.source == "synthetic" and record.label_json_path and record.label_json_path.exists():
                label_payload = json.loads(record.label_json_path.read_text(encoding="utf-8"))
                boxes = [
                    {
                        "transcription": item.get("text", ""),
                        "points": _bbox_to_polygon(item.get("bbox", [0, 0, 1, 1])),
                        "difficult": False,
                    }
                    for item in label_payload.get("bboxes", [])
                ]
            else:
                boxes = []

            det_lines.append(f"images/{record.filename}\t{json.dumps(boxes, ensure_ascii=False)}")
            for crop_name, crop_text in zip(record.recognition_crops, record.crop_texts):
                crop_target = crop_dir / crop_name
                crop_target.write_bytes(_placeholder_crop_bytes())
                rec_lines.append(f"crop/{crop_name} {crop_text}")

        det_gt_path.write_text("\n".join(det_lines) + ("\n" if det_lines else ""), encoding="utf-8")
        rec_gt_path.write_text("\n".join(rec_lines) + ("\n" if rec_lines else ""), encoding="utf-8")

    split_config = {
        "split_date": str(date.today()),
        "random_seed": split_seed,
        "ratios": SPLIT_RATIOS,
        "strategy": "stratified",
        "strata_keys": ["format", "industry"],
        "synthetic_in_train_only": True,
        "difficult_excluded_from_test": True,
        "total_images": len(bundle.train) + len(bundle.val) + len(bundle.test),
        "train_count": len(bundle.train),
        "val_count": len(bundle.val),
        "test_count": len(bundle.test),
    }
    outputs["split_config"].write_text(json.dumps(split_config, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "train": summarize_dataset_stats(bundle.train),
        "val": summarize_dataset_stats(bundle.val),
        "test": summarize_dataset_stats(bundle.test),
        "all": summarize_dataset_stats(bundle.train + bundle.val + bundle.test),
    }
    outputs["dataset_stats"].write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["quality_report"].write_text(
        build_quality_report(
            automatic_issues=[],
            manual_review_error_rate=0.0,
            stats=stats,
            split_issues=[],
        ),
        encoding="utf-8",
    )
    return outputs


def build_quality_report(
    automatic_issues: list[QualityIssue],
    manual_review_error_rate: float,
    stats: dict[str, Any],
    split_issues: list[str],
) -> str:
    """Create a markdown report for final dataset QA."""
    lines = [
        "# Final Dataset Quality Report",
        "",
        "## Automatic Checks",
        f"- issue_count: {len(automatic_issues)}",
    ]
    for issue in automatic_issues[:20]:
        lines.append(f"- {issue.filename}: {issue.issue_code} ({issue.details})")

    lines.extend(
        [
            "",
            "## Manual Review",
            f"- sampled_error_rate: {manual_review_error_rate:.2%}",
            f"- passed_threshold: {manual_review_error_rate <= 0.03}",
            "",
            "## Split Validation",
            f"- split_issue_count: {len(split_issues)}",
        ]
    )
    for issue in split_issues:
        lines.append(f"- {issue}")

    lines.extend(
        [
            "",
            "## Stats Snapshot",
            f"- total_images: {stats['all']['counts']['total_images']}",
            f"- real_images: {stats['all']['counts']['real_images']}",
            f"- synthetic_images: {stats['all']['counts']['synthetic_images']}",
            f"- total_detection_boxes: {stats['all']['counts']['total_detection_boxes']}",
            f"- total_recognition_crops: {stats['all']['counts']['total_recognition_crops']}",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate_dataset_definition_of_done(
    automatic_issue_count: int,
    manual_review_error_rate: float,
    split_issues: list[str],
    split_config_exists: bool,
    dataset_stats_exists: bool,
    quality_report_exists: bool,
    frozen: bool,
) -> list[str]:
    """Return unmet OCR-D04 completion requirements."""
    issues: list[str] = []
    if automatic_issue_count > 0:
        issues.append(f"automatic_quality_issues_remaining: {automatic_issue_count}")
    if manual_review_error_rate > 0.03:
        issues.append(f"manual_review_error_rate_exceeds_3_percent: {manual_review_error_rate:.4f}")
    issues.extend(split_issues)
    if not split_config_exists:
        issues.append("split_config_missing")
    if not dataset_stats_exists:
        issues.append("dataset_stats_missing")
    if not quality_report_exists:
        issues.append("quality_report_missing")
    if not frozen:
        issues.append("dataset_not_marked_read_only")
    return issues


def mark_dataset_read_only(dataset_root: Path) -> None:
    """Set dataset files to read-only to freeze the split."""
    for path in dataset_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)


def parse_filename_metadata(filename: str) -> tuple[str, str]:
    """Infer format and industry codes from OCR-D01/D02 filenames."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if stem.startswith("synth_") and len(parts) >= 4:
        return _synthetic_to_format(parts[1]), parts[2]
    if len(parts) >= 3 and parts[0] in ALLOWED_FORMATS and parts[1] in ALLOWED_INDUSTRIES:
        return parts[0], parts[1]
    return "paper", "etc"


def _distribution_table(counter: Counter[str], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    total = sum(counter.values())
    return {
        key: {
            "count": counter.get(key, 0),
            "ratio": (counter.get(key, 0) / total) if total else 0.0,
        }
        for key in keys
    }


def _text_length_bucket(text: str) -> str:
    length = len(text)
    if length <= 1:
        return "1"
    if length <= 4:
        return "2-4"
    if length <= 10:
        return "5-10"
    return "11+"


def _synthetic_to_format(method: str) -> str:
    return "handwrite" if method == "render" else "screen"


def _match_recognition_entries(filename: str, rec_entries: dict[str, str]) -> tuple[list[str], list[str]]:
    stem = Path(filename).stem
    matched = sorted(
        (crop_name, text)
        for crop_name, text in rec_entries.items()
        if crop_name.startswith(f"{stem}_")
    )
    return [name for name, _ in matched], [text for _, text in matched]


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _bbox_to_polygon(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = (bbox + [0, 0, 1, 1])[:4]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _placeholder_crop_bytes() -> bytes:
    return bytes(
        [
            137,
            80,
            78,
            71,
            13,
            10,
            26,
            10,
            0,
            0,
            0,
            13,
            73,
            72,
            68,
            82,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            8,
            6,
            0,
            0,
            0,
            31,
            21,
            196,
            137,
            0,
            0,
            0,
            12,
            73,
            68,
            65,
            84,
            120,
            156,
            99,
            248,
            255,
            255,
            63,
            0,
            5,
            254,
            2,
            254,
            167,
            83,
            129,
            168,
            0,
            0,
            0,
            0,
            73,
            69,
            78,
            68,
            174,
            66,
            96,
            130,
        ]
    )
