from __future__ import annotations

import base64
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ocr_project.stage1_data.collection_spec import ALLOWED_INDUSTRIES, CollectionSpec, default_collection_spec


RENDER_TARGET_MIN = 150
AUGMENT_TARGET_MIN = 200
HANDWRITING_TARGET_MIN = 100
TOTAL_SYNTHETIC_RANGE = (400, 600)
MAX_SYNTHETIC_RATIO = 0.5
MAX_AUG_PER_SOURCE = 10

RENDER_METHOD = "render"
AUG_METHOD = "aug"
METHOD_CODES: tuple[str, ...] = (RENDER_METHOD, AUG_METHOD)
PLACEHOLDER_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9Wn4nS4AAAAASUVORK5CYII="
)

TEMPLATE_TYPES: tuple[str, ...] = (
    "monthly_calendar",
    "weekly_block",
    "name_code_list",
)

PRINT_FONTS: tuple[str, ...] = (
    "NanumGothic",
    "NanumMyeongjo",
    "KoPubWorldDotum",
    "KoPubWorldBatang",
    "MalgunGothic",
    "AppleGothic",
    "NotoSansKR",
)

HANDWRITING_FONTS: tuple[str, ...] = (
    "NanumPen",
    "NanumBrush",
    "GanaGreenUmbrella",
    "Cafe24Oneprettynight",
    "KyoboHand",
    "SCoreDreamHandwriting",
)

LATIN_FONTS: tuple[str, ...] = ("Arial", "Helvetica", "Roboto")
ALL_FONTS: tuple[str, ...] = PRINT_FONTS + HANDWRITING_FONTS + LATIN_FONTS

WORK_CODE_LEXICON: dict[str, list[str]] = {
    "korean": [
        "낮",
        "밤",
        "오전",
        "오후",
        "저녁",
        "야간",
        "주간",
        "오프",
        "휴무",
        "반차",
        "연차",
        "당직",
        "재택",
        "출장",
        "조기퇴근",
        "대휴",
        "보상휴가",
    ],
    "english": ["D", "E", "N", "A", "B", "C", "OFF", "AM", "PM", "OT", "AL", "SL"],
    "mixed": ["1", "2", "3", "4", "1조", "2조", "A조", "B조"],
}

NAME_LEXICON: dict[str, list[str]] = {
    "family": ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"],
    "given": ["민준", "서연", "지우", "예진", "현우", "수빈", "지원", "태양", "나연", "도현"],
    "aliases": ["김M", "이T", "박간호사", "J.Kim", "S.Lee"],
}

DATE_LEXICON: dict[str, list[str]] = {
    "dates": [f"{day}일" for day in range(1, 32)] + [f"{month}/{day}" for month in range(1, 13) for day in (1, 15, 28)],
    "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "월", "화", "수", "목", "금", "토", "일"],
    "times": ["09:00", "18:00", "22:00~06:00", "08:30~17:30"],
}

RENDER_NOISE_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "gaussian_noise", "probability": 0.70, "range": "sigma=1~5"},
    {"name": "ink_bleed", "probability": 0.30, "range": "kernel=1~3"},
    {"name": "ink_break", "probability": 0.30, "range": "kernel=1~2"},
    {"name": "paper_texture_overlay", "probability": 0.50, "range": "alpha=10~30%"},
    {"name": "line_shadow", "probability": 0.20, "range": "strength=low~medium"},
)

AUGMENTATION_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "rotation", "probability": 0.80, "range": "-30~30deg", "strength": 0.25},
    {"name": "perspective", "probability": 0.50, "range": "max_shift=10%", "strength": 0.20},
    {"name": "scale_x", "probability": 0.40, "range": "0.8~1.2", "strength": 0.10},
    {"name": "scale_y", "probability": 0.40, "range": "0.8~1.2", "strength": 0.10},
    {"name": "crop", "probability": 0.30, "range": "70~95% area", "strength": 0.15},
    {"name": "brightness", "probability": 0.80, "range": "0.5~1.5", "strength": 0.10},
    {"name": "contrast", "probability": 0.70, "range": "0.6~1.8", "strength": 0.10},
    {"name": "motion_blur", "probability": 0.20, "range": "kernel=3~7", "strength": 0.10},
    {"name": "gaussian_blur", "probability": 0.30, "range": "sigma=0.5~1.5", "strength": 0.08},
    {"name": "jpeg_artifact", "probability": 0.60, "range": "quality=40~85", "strength": 0.06},
    {"name": "shadow_overlay", "probability": 0.25, "range": "single_side_shadow", "strength": 0.08},
)

SYNTH_LOG_COLUMNS: tuple[str, ...] = (
    "filename",
    "method",
    "industry",
    "source_filename",
    "template_type",
    "font",
    "is_handwriting",
    "label_path",
    "passed_quality_check",
)


@dataclass(slots=True)
class SyntheticGenerationRequest:
    template_dir: Path
    output_dir: Path
    label_dir: Path
    count: int
    method: str
    industry: str
    seed: int = 42
    source_images: list[Path] = field(default_factory=list)
    source_labels: list[Path] = field(default_factory=list)
    handwriting_ratio: float = 0.25


@dataclass(slots=True)
class SyntheticBBox:
    text: str
    bbox: list[int]
    font: str
    font_size: int


@dataclass(slots=True)
class SyntheticLabel:
    filename: str
    method: str
    industry: str
    bboxes: list[SyntheticBBox]
    augmentations_applied: list[dict[str, Any]]
    source_label_path: str | None = None
    source_image_path: str | None = None
    template_type: str | None = None


@dataclass(slots=True)
class SyntheticPlan:
    total_target: int = 400
    render_minimum: int = RENDER_TARGET_MIN
    augment_minimum: int = AUGMENT_TARGET_MIN
    handwriting_minimum: int = HANDWRITING_TARGET_MIN
    max_synthetic_ratio: float = MAX_SYNTHETIC_RATIO
    max_augmented_per_source: int = MAX_AUG_PER_SOURCE
    total_range: tuple[int, int] = TOTAL_SYNTHETIC_RANGE


@dataclass(slots=True)
class GapSummary:
    real_count: int
    recommended_render: int
    recommended_aug: int
    reason: str


def initialize_synthetic_workspace(root_dir: Path) -> dict[str, Path]:
    """Create OCR-D02 folders and metadata files."""
    directories = [
        root_dir / "data" / "synthetic",
        root_dir / "data" / "synthetic" / "render",
        root_dir / "data" / "synthetic" / "aug",
        root_dir / "data" / "synthetic" / "labels",
        root_dir / "data" / "meta",
    ]

    for method in METHOD_CODES:
        for industry in _synthetic_industries():
            directories.append(root_dir / "data" / "synthetic" / method / industry)

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    synth_log = root_dir / "data" / "meta" / "synth_log.csv"
    synth_params = root_dir / "data" / "meta" / "synth_params.json"
    _ensure_csv_header(synth_log, SYNTH_LOG_COLUMNS)
    if not synth_params.exists():
        synth_params.write_text(
            json.dumps(default_synth_params(root_dir), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "synthetic_root": root_dir / "data" / "synthetic",
        "render_root": root_dir / "data" / "synthetic" / "render",
        "aug_root": root_dir / "data" / "synthetic" / "aug",
        "label_root": root_dir / "data" / "synthetic" / "labels",
        "synth_log": synth_log,
        "synth_params": synth_params,
    }


def plan_supplement_counts(real_counts: dict[str, int]) -> dict[str, GapSummary]:
    """Decide how many synthetic images to generate for each industry/format bucket."""
    plan: dict[str, GapSummary] = {}
    for bucket_name, real_count in real_counts.items():
        if real_count <= 0:
            plan[bucket_name] = GapSummary(
                real_count=real_count,
                recommended_render=80,
                recommended_aug=0,
                reason="no_real_samples_use_render_only",
            )
        elif real_count < 50:
            plan[bucket_name] = GapSummary(
                real_count=real_count,
                recommended_render=25,
                recommended_aug=25,
                reason="low_real_samples_use_mixed_top_up",
            )
        else:
            plan[bucket_name] = GapSummary(
                real_count=real_count,
                recommended_render=0,
                recommended_aug=min(real_count * 3, 200),
                reason="sufficient_real_samples_aug_only",
            )
    return plan


def build_synth_filename(method: str, industry: str, sequence: int) -> str:
    """Generate the OCR-D02 synthetic filename."""
    if method not in METHOD_CODES:
        raise ValueError(f"unsupported synthesis method: {method}")
    if industry not in _synthetic_industries():
        raise ValueError(f"unsupported industry: {industry}")
    return f"synth_{method}_{industry}_{sequence:05d}.png"


def generate_synthetic_samples(request: SyntheticGenerationRequest) -> list[Path]:
    """Create placeholder synthetic outputs, labels, and metadata rows."""
    request.output_dir.mkdir(parents=True, exist_ok=True)
    request.label_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(request.seed)
    created_files: list[Path] = []

    for sequence in range(1, request.count + 1):
        filename = build_synth_filename(request.method, request.industry, sequence)
        output_path = request.output_dir / filename

        if request.method == RENDER_METHOD:
            label = _build_render_label(filename, request.industry, rng, request.handwriting_ratio)
        else:
            label = _build_aug_label(filename, request, rng, sequence)

        output_path.write_bytes(base64.b64decode(PLACEHOLDER_PNG_BASE64))
        label_path = request.label_dir / f"{Path(filename).stem}.json"
        label_path.write_text(
            json.dumps(_label_to_dict(label), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        created_files.append(output_path)

    return created_files


def append_synth_log(
    csv_path: Path,
    *,
    filename: str,
    method: str,
    industry: str,
    source_filename: str,
    template_type: str,
    font: str,
    is_handwriting: bool,
    label_path: str,
    passed_quality_check: bool,
) -> None:
    _append_csv_row(
        csv_path,
        {
            "filename": filename,
            "method": method,
            "industry": industry,
            "source_filename": source_filename,
            "template_type": template_type,
            "font": font,
            "is_handwriting": "Y" if is_handwriting else "N",
            "label_path": label_path,
            "passed_quality_check": "Y" if passed_quality_check else "N",
        },
    )


def default_synth_params(root_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Serialize OCR-D02 defaults for reproducibility."""
    spec: CollectionSpec = default_collection_spec()
    return {
        "seed": seed,
        "synthetic_plan": asdict(SyntheticPlan()),
        "collection_quality_thresholds": asdict(spec.quality),
        "template_types": list(TEMPLATE_TYPES),
        "print_fonts": list(PRINT_FONTS),
        "handwriting_fonts": list(HANDWRITING_FONTS),
        "latin_fonts": list(LATIN_FONTS),
        "work_code_lexicon": WORK_CODE_LEXICON,
        "name_lexicon": NAME_LEXICON,
        "date_lexicon": DATE_LEXICON,
        "render_noise_specs": list(RENDER_NOISE_SPECS),
        "augmentation_specs": list(AUGMENTATION_SPECS),
        "paths": {
            "masked": str(root_dir / "data" / "masked"),
            "synthetic": str(root_dir / "data" / "synthetic"),
            "labels": str(root_dir / "data" / "synthetic" / "labels"),
        },
        "notes": {
            "synthetic_ratio_max": MAX_SYNTHETIC_RATIO,
            "font_cap_ratio": 0.20,
            "manual_visual_qc_min_ratio": 0.05,
        },
    }


def evaluate_synthetic_definition_of_done(
    render_count: int,
    aug_count: int,
    handwriting_render_count: int,
    label_count: int,
    synth_log_exists: bool,
    synth_params_exists: bool,
    visual_qc_passed: bool,
    masked_count: int,
) -> list[str]:
    """Return unmet OCR-D02 completion requirements."""
    issues: list[str] = []
    total_synth = render_count + aug_count
    total_dataset = total_synth + masked_count

    if total_synth < TOTAL_SYNTHETIC_RANGE[0]:
        issues.append(f"synthetic_total_below_minimum: {total_synth} < {TOTAL_SYNTHETIC_RANGE[0]}")
    if render_count < RENDER_TARGET_MIN:
        issues.append(f"render_count_below_minimum: {render_count} < {RENDER_TARGET_MIN}")
    if aug_count < AUGMENT_TARGET_MIN:
        issues.append(f"aug_count_below_minimum: {aug_count} < {AUGMENT_TARGET_MIN}")
    if handwriting_render_count < HANDWRITING_TARGET_MIN:
        issues.append(
            f"handwriting_render_count_below_minimum: {handwriting_render_count} < {HANDWRITING_TARGET_MIN}"
        )
    if label_count < total_synth:
        issues.append(f"missing_label_jsons: {label_count} < {total_synth}")
    if not synth_log_exists:
        issues.append("synth_log_csv_missing")
    if not synth_params_exists:
        issues.append("synth_params_json_missing")
    if not visual_qc_passed:
        issues.append("visual_qc_not_passed")
    if total_dataset < 600:
        issues.append(f"combined_dataset_below_600: {total_dataset} < 600")
    if total_dataset > 0 and total_synth / total_dataset > MAX_SYNTHETIC_RATIO:
        issues.append("synthetic_ratio_exceeds_50_percent")

    return issues


def recommend_visual_qc_sample_size(total_generated: int) -> int:
    """At least 5 percent of synthetic images should be manually reviewed."""
    return max(1, int(total_generated * 0.05))


def _build_render_label(
    filename: str,
    industry: str,
    rng: random.Random,
    handwriting_ratio: float,
) -> SyntheticLabel:
    use_handwriting = rng.random() < handwriting_ratio
    font = rng.choice(HANDWRITING_FONTS if use_handwriting else ALL_FONTS)
    template_type = rng.choice(TEMPLATE_TYPES)
    font_size = rng.randint(12, 20)
    texts = [
        _sample_name(rng),
        rng.choice(WORK_CODE_LEXICON["korean"] + WORK_CODE_LEXICON["english"] + WORK_CODE_LEXICON["mixed"]),
        rng.choice(DATE_LEXICON["dates"]),
    ]
    bboxes = [
        SyntheticBBox(text=text, bbox=[10 + idx * 60, 10, 50 + idx * 60, 30], font=font, font_size=font_size)
        for idx, text in enumerate(texts)
    ]
    augmentations_applied = _sample_render_noise(rng)
    return SyntheticLabel(
        filename=filename,
        method=RENDER_METHOD,
        industry=industry,
        bboxes=bboxes,
        augmentations_applied=augmentations_applied,
        template_type=template_type,
    )


def _build_aug_label(
    filename: str,
    request: SyntheticGenerationRequest,
    rng: random.Random,
    sequence: int,
) -> SyntheticLabel:
    source_image = request.source_images[(sequence - 1) % len(request.source_images)] if request.source_images else None
    source_label = request.source_labels[(sequence - 1) % len(request.source_labels)] if request.source_labels else None
    augmentations = _sample_augmentations(rng)
    return SyntheticLabel(
        filename=filename,
        method=AUG_METHOD,
        industry=request.industry,
        bboxes=[],
        augmentations_applied=augmentations,
        source_label_path=str(source_label) if source_label else None,
        source_image_path=str(source_image) if source_image else None,
    )


def _sample_name(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.25:
        return rng.choice(NAME_LEXICON["aliases"])
    return f"{rng.choice(NAME_LEXICON['family'])}{rng.choice(NAME_LEXICON['given'])}"


def _sample_render_noise(rng: random.Random) -> list[dict[str, Any]]:
    return [spec for spec in RENDER_NOISE_SPECS if rng.random() <= spec["probability"]]


def _sample_augmentations(rng: random.Random) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    total_strength = 0.0
    for spec in AUGMENTATION_SPECS:
        if rng.random() <= spec["probability"]:
            if total_strength + spec["strength"] > 0.65:
                continue
            sampled.append(spec)
            total_strength += float(spec["strength"])
    return sampled


def _synthetic_industries() -> tuple[str, ...]:
    return tuple(industry for industry in ALLOWED_INDUSTRIES if industry != "etc")


def _label_to_dict(label: SyntheticLabel) -> dict[str, Any]:
    return {
        "filename": label.filename,
        "method": label.method,
        "industry": label.industry,
        "bboxes": [
            {
                "text": bbox.text,
                "bbox": bbox.bbox,
                "font": bbox.font,
                "font_size": bbox.font_size,
            }
            for bbox in label.bboxes
        ],
        "augmentations_applied": label.augmentations_applied,
        "source_label_path": label.source_label_path,
        "source_image_path": label.source_image_path,
        "template_type": label.template_type,
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
