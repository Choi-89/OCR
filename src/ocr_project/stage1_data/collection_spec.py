from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


FORMAT_TARGETS: dict[str, int] = {
    "paper": 200,
    "scan": 100,
    "screen": 150,
    "excel": 100,
    "handwrite": 50,
}

INDUSTRY_MINIMUMS: dict[str, int] = {
    "hospital": 50,
    "convenience": 50,
    "factory": 50,
    "office": 50,
    "food": 50,
}

ALLOWED_FORMATS: tuple[str, ...] = tuple(FORMAT_TARGETS)
ALLOWED_INDUSTRIES: tuple[str, ...] = (
    "hospital",
    "convenience",
    "factory",
    "office",
    "food",
    "etc",
)
ALLOWED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".pdf")


@dataclass(slots=True)
class QualityThreshold:
    min_short_side_px: int = 720
    recommended_short_side_px: int = 1080
    max_tilt_degrees: int = 30
    recommended_tilt_degrees: int = 15
    min_text_area_ratio: float = 0.20
    min_legibility_ratio: float = 0.80
    recommended_legibility_ratio: float = 0.95
    min_file_size_kb: int = 100
    max_file_size_mb: int = 20
    recommended_file_size_min_kb: int = 500
    recommended_file_size_max_mb: int = 5


@dataclass(slots=True)
class PrivacyPolicy:
    mask_targets: tuple[str, ...] = (
        "employee_name",
        "employee_id",
        "phone_number",
        "email",
        "organization_name",
        "department_name",
    )
    destination_dir: str = "data/masked"
    quarantine_dir: str = "data/raw"
    rejected_dir: str = "data/rejected"


@dataclass(slots=True)
class CollectionSpec:
    work_type: str
    total_min_samples: int
    format_targets: dict[str, int] = field(default_factory=lambda: dict(FORMAT_TARGETS))
    industry_minimums: dict[str, int] = field(default_factory=lambda: dict(INDUSTRY_MINIMUMS))
    file_extensions: tuple[str, ...] = ALLOWED_EXTENSIONS
    quality: QualityThreshold = field(default_factory=QualityThreshold)
    privacy: PrivacyPolicy = field(default_factory=PrivacyPolicy)

    def supports_format(self, format_code: str) -> bool:
        return format_code in ALLOWED_FORMATS

    def supports_industry(self, industry_code: str) -> bool:
        return industry_code in ALLOWED_INDUSTRIES


def default_collection_spec() -> CollectionSpec:
    """OCR-D01: Define collection rules for payroll document images."""
    return CollectionSpec(
        work_type="shiftflow_payroll_schedule",
        total_min_samples=600,
    )


def expected_data_directories(root_dir: Path) -> list[Path]:
    """Return the required OCR-D01 directory layout."""
    return [
        root_dir / "data",
        root_dir / "data" / "raw",
        root_dir / "data" / "raw" / "paper",
        root_dir / "data" / "raw" / "scan",
        root_dir / "data" / "raw" / "screen",
        root_dir / "data" / "raw" / "excel",
        root_dir / "data" / "raw" / "handwrite",
        root_dir / "data" / "masked",
        root_dir / "data" / "rejected",
        root_dir / "data" / "meta",
    ]
