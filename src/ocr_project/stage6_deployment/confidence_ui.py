from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ConfidenceConfig:
    high_threshold: float = 0.90
    low_threshold: float = 0.70


@dataclass(slots=True)
class OCRFeedbackRequest:
    import_id: str
    cell_id: str
    original_text: str
    corrected_text: str
    confidence: float
    confidence_level: str
    image_crop_path: str | None = None


@dataclass(slots=True)
class FeedbackStoreConfig:
    root: Path = Path("data/feedback")
    log_name: str = "feedback_log.jsonl"
    crops_dir_name: str = "crops"


def confidence_ui_fields() -> list[str]:
    """OCR-S03: UI bindings for OCR confidence and feedback."""
    return ["confidence", "confidence_level", "cell_id", "summary", "feedback"]


def confidence_config_from_env() -> ConfidenceConfig:
    return ConfidenceConfig(
        high_threshold=float(os.getenv("OCR_CONFIDENCE_HIGH", "0.90")),
        low_threshold=float(os.getenv("OCR_CONFIDENCE_LOW", "0.70")),
    )


def confidence_level(score: float, config: ConfidenceConfig | None = None) -> str:
    cfg = config or ConfidenceConfig()
    if score >= cfg.high_threshold:
        return "high"
    if score < cfg.low_threshold:
        return "low"
    return "mid"


def build_cell_id(import_id: str, index: int, row: int | None = None, col: int | None = None) -> str:
    if row is not None and col is not None:
        return f"{import_id}_row{row:02d}_col{col:02d}"
    return f"{import_id}_cell{index:03d}"


def enrich_ocr_response(
    response: dict[str, Any],
    *,
    import_id: str = "imp_pending",
    config: ConfidenceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ConfidenceConfig()
    results = []
    for index, item in enumerate(response.get("results", []), start=1):
        confidence = float(item.get("confidence", item.get("score", 0.0)))
        enriched = dict(item)
        enriched["confidence"] = round(confidence, 4)
        enriched["confidence_level"] = confidence_level(confidence, cfg)
        enriched["cell_id"] = item.get("cell_id") or build_cell_id(import_id, index)
        results.append(enriched)
    output = dict(response)
    output["results"] = results
    output["summary"] = confidence_summary(results)
    return output


def confidence_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(item.get("confidence_level", "low") for item in results)
    mid = counts.get("mid", 0)
    low = counts.get("low", 0)
    return {
        "total_cells": len(results),
        "high_confidence": counts.get("high", 0),
        "mid_confidence": mid,
        "low_confidence": low,
        "review_required": mid + low,
    }


def submit_feedback(
    request: OCRFeedbackRequest,
    *,
    store_config: FeedbackStoreConfig | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    cfg = store_config or FeedbackStoreConfig()
    cfg.root.mkdir(parents=True, exist_ok=True)
    feedback_id = generate_feedback_id(cfg.root / cfg.log_name)
    record = {
        "feedback_id": feedback_id,
        "import_id": request.import_id,
        "cell_id": request.cell_id,
        "worker_id": worker_id,
        "original_text": request.original_text,
        "corrected_text": request.corrected_text,
        "confidence": request.confidence,
        "confidence_level": request.confidence_level,
        "timestamp": now_iso(),
        "crop_saved": False,
    }
    if request.image_crop_path:
        crop_path = save_feedback_crop(request, cfg)
        if crop_path:
            record["crop_saved"] = True
            record["crop_path"] = str(crop_path)
    append_jsonl(cfg.root / cfg.log_name, record)
    return {"feedback_id": feedback_id, "status": "recorded"}


def save_feedback_crop(request: OCRFeedbackRequest, cfg: FeedbackStoreConfig) -> Path | None:
    source = Path(request.image_crop_path or "")
    if not source.exists():
        return None
    target_dir = cfg.root / cfg.crops_dir_name / request.import_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{request.cell_id}{source.suffix or '.png'}"
    shutil.copyfile(source, target)
    return target


def feedback_stats(
    feedback_log: str | Path,
    *,
    retrain_threshold: int = 100,
) -> dict[str, Any]:
    records = read_feedback_log(feedback_log)
    usable = [record for record in records if feedback_is_usable(record)]
    corrections = Counter((record.get("original_text", ""), record.get("corrected_text", "")) for record in records)
    top_corrections = [
        {"original": original, "corrected": corrected, "count": count}
        for (original, corrected), count in corrections.most_common(10)
        if original != corrected
    ]
    by_level = Counter(record.get("confidence_level", "unknown") for record in records)
    last = max((record.get("timestamp", "") for record in records), default="")
    return {
        "total_feedback": len(records),
        "usable_feedback": len(usable),
        "by_confidence_level": dict(by_level),
        "top_corrections": top_corrections,
        "last_feedback_date": last[:10] if last else "",
        "ready_for_retrain": len(usable) >= retrain_threshold,
        "retrain_threshold": retrain_threshold,
    }


def build_feedback_dataset(
    feedback_log: str | Path,
    crops_dir: str | Path,
    output_dir: str | Path,
    *,
    min_feedback_count: int = 50,
    dictionary: set[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    crop_output = output / "crop"
    crop_output.mkdir(parents=True, exist_ok=True)
    records = [record for record in read_feedback_log(feedback_log) if feedback_is_usable(record, dictionary=dictionary)]
    if len(records) < min_feedback_count:
        stats = {"created": False, "reason": "not_enough_feedback", "usable_feedback": len(records), "min_feedback_count": min_feedback_count}
        write_json(output / "stats.json", stats)
        return stats
    rec_lines: list[str] = []
    for record in records:
        crop_path = Path(record.get("crop_path", ""))
        if not crop_path.exists():
            fallback = Path(crops_dir) / record["import_id"] / f"{record['cell_id']}.png"
            crop_path = fallback
        if not crop_path.exists():
            continue
        target = crop_output / crop_path.name
        shutil.copyfile(crop_path, target)
        rec_lines.append(f"crop/{target.name} {record['corrected_text']}")
    (output / "rec_gt.txt").write_text("\n".join(rec_lines) + ("\n" if rec_lines else ""), encoding="utf-8")
    stats = {"created": True, "usable_feedback": len(records), "exported_crops": len(rec_lines), "excluded": len(read_feedback_log(feedback_log)) - len(records)}
    write_json(output / "stats.json", stats)
    return stats


def feedback_is_usable(record: dict[str, Any], dictionary: set[str] | None = None) -> bool:
    original = str(record.get("original_text", ""))
    corrected = str(record.get("corrected_text", ""))
    if original == corrected:
        return False
    if not (1 <= len(corrected) <= 20):
        return False
    if not record.get("crop_saved", False):
        return False
    if dictionary and any(char not in dictionary for char in corrected):
        return False
    return True


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_feedback_log(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_feedback_id(log_path: Path) -> str:
    next_index = len(read_feedback_log(log_path)) + 1
    return f"fb_{next_index:06d}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def feedback_request_from_dict(payload: dict[str, Any]) -> OCRFeedbackRequest:
    return OCRFeedbackRequest(**payload)


def feedback_request_to_dict(request: OCRFeedbackRequest) -> dict[str, Any]:
    return asdict(request)
