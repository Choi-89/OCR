from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ToolCandidate:
    name: str
    advantages: tuple[str, ...]
    limitations: tuple[str, ...]
    fitness: str


@dataclass(slots=True)
class AnnotationRule:
    field_name: str
    label_name: str
    description: str


@dataclass(slots=True)
class BoundingBoxRule:
    scenario: str
    handling: str


@dataclass(slots=True)
class TextInputRule:
    scenario: str
    handling: str


@dataclass(slots=True)
class AnnotationPolicy:
    primary_tool: str = "PPOCRLabel"
    fallback_tool: str = "LabelStudio"
    target_dir: str = "data/masked"
    detection_output: str = "data/labels/det_gt.txt"
    recognition_output: str = "data/labels/rec_gt.txt"
    crop_dir: str = "data/labels/crop"
    difficult_dir: str = "data/labels/difficult"
    unreadable_dir: str = "data/labels/unreadable"
    annotation_log: str = "data/meta/annotation_log.csv"
    calibration_dir: str = "data/meta/calibration"
    calibration_image_count: int = 10
    cross_review_ratio: float = 0.10
    min_mean_iou: float = 0.85
    max_unreadable_ratio: float = 0.05
    difficult_token: str = "difficult"
    unreadable_token: str = "###"
    polygon_angle_threshold_degrees: int = 15
    bbox_padding_px: int = 2
    review_batch_size: int = 100


def recommended_tools() -> list[ToolCandidate]:
    """OCR-D03: Compare annotation tool candidates."""
    return [
        ToolCandidate(
            name="LabelImg",
            advantages=("easy_install", "bbox_focused"),
            limitations=("text_input_ui_poor", "maintenance_stalled"),
            fitness="low",
        ),
        ToolCandidate(
            name="LabelStudio",
            advantages=("web_based", "bbox_and_text", "team_collaboration"),
            limitations=("initial_setup_complex"),
            fitness="high",
        ),
        ToolCandidate(
            name="PPOCRLabel",
            advantages=("paddleocr_native_format", "auto_recognition_assist", "local_only"),
            limitations=("initial_korean_accuracy_limited"),
            fitness="very_high",
        ),
        ToolCandidate(
            name="Roboflow",
            advantages=("intuitive_ui", "cloud_sharing"),
            limitations=("free_tier_limits", "external_upload_required"),
            fitness="medium",
        ),
    ]


def build_annotation_rules() -> list[AnnotationRule]:
    """Core semantic labels used during OCR annotation."""
    return [
        AnnotationRule("employee_name", "NAME", "Keep the visible employee name as-is."),
        AnnotationRule("work_code", "WORK_CODE", "Label one code token per text unit."),
        AnnotationRule("date_text", "DATE", "Keep the visible date format unchanged."),
        AnnotationRule("time_text", "TIME", "Keep the visible time format unchanged."),
    ]


def bbox_rules() -> list[BoundingBoxRule]:
    """Bounding-box drawing rules from OCR-D03."""
    return [
        BoundingBoxRule("single_code_in_cell", "draw_one_box_with_2px_padding"),
        BoundingBoxRule("code_and_time_in_one_cell", "split_into_two_boxes"),
        BoundingBoxRule("name_in_single_cell", "draw_one_box"),
        BoundingBoxRule("date_row", "draw_each_date_number_separately"),
        BoundingBoxRule("empty_cell", "skip_without_box"),
        BoundingBoxRule("multiline_text", "split_per_line"),
        BoundingBoxRule("tilt_over_15_degrees", "use_polygon"),
        BoundingBoxRule("tilt_15_degrees_or_less", "axis_aligned_allowed"),
        BoundingBoxRule("masked_region", "skip_without_box"),
    ]


def text_input_rules() -> list[TextInputRule]:
    """Visible-text transcription rules from OCR-D03."""
    return [
        TextInputRule("mixed_korean_english", "transcribe_exactly_as_visible"),
        TextInputRule("slash_separated_text", "keep_as_single_text"),
        TextInputRule("date_expression", "preserve_original_format"),
        TextInputRule("time_expression", "preserve_original_format"),
        TextInputRule("text_with_parentheses", "include_parentheses"),
        TextInputRule("underlined_or_struck_text", "ignore_style_keep_text_only"),
        TextInputRule("unreadable_text", "use_###"),
        TextInputRule("do_not_normalize_case", "never_modify_original_characters"),
    ]


def default_annotation_policy() -> AnnotationPolicy:
    """Return the default annotation operating policy."""
    return AnnotationPolicy()
