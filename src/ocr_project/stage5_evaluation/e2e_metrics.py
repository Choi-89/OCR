from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol


BBox = list[float]


@dataclass(slots=True)
class OCRToken:
    text: str
    bbox: BBox
    score: float = 1.0

    @property
    def x_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])


@dataclass(slots=True)
class ScheduleMatrix:
    image_path: str
    format: str
    industry: str
    year_month: str
    workers: list[str]
    schedule: dict[str, dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cells(self) -> int:
        return sum(len(days) for days in self.schedule.values())


@dataclass(slots=True)
class ParseConfig:
    row_height_ratio: float = 0.5
    col_width_ratio: float = 0.5
    min_header_numeric_ratio: float = 0.5


class OCRPipeline(Protocol):
    def run(self, image_path: str | Path) -> list[dict[str, Any]]:
        ...


@dataclass(slots=True)
class E2EImageResult:
    image_path: str
    parse_success: bool
    cell_accuracy: float
    worker_schedule_accuracy: float
    name_accuracy: float
    code_distribution_error: float
    total_cells: int
    correct_cells: int
    errors: list[dict[str, str]]
    format: str
    industry: str


@dataclass(slots=True)
class E2EEvaluator:
    gt_dir: str | Path
    parse_config: ParseConfig = field(default_factory=ParseConfig)

    def load_ground_truth(self, image_path: str | Path) -> ScheduleMatrix:
        gt_path = Path(self.gt_dir) / f"{Path(image_path).stem}.json"
        payload = json.loads(gt_path.read_text(encoding="utf-8"))
        return ScheduleMatrix(
            image_path=payload["image_path"],
            format=payload.get("format", "unknown"),
            industry=payload.get("industry", "unknown"),
            year_month=payload.get("year_month", ""),
            workers=list(payload.get("workers", [])),
            schedule=payload.get("schedule", {}),
            metadata=payload.get("metadata", {}),
        )

    def evaluate_image(self, image_path: str | Path, ocr_pipeline: OCRPipeline | list[dict[str, Any]]) -> dict[str, Any]:
        gt = self.load_ground_truth(image_path)
        tokens = run_or_wrap_ocr(image_path, ocr_pipeline)
        pred, parse_success = parse_schedule_from_tokens(tokens, gt, self.parse_config)
        result = compare_schedule(gt, pred, parse_success)
        return asdict(result)

    def evaluate_dataset(self, image_dir: str | Path, ocr_pipeline: OCRPipeline | dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        failed_parse: list[str] = []
        for gt_path in sorted(Path(self.gt_dir).glob("*.json")):
            image_name = json.loads(gt_path.read_text(encoding="utf-8")).get("image_path", gt_path.with_suffix("").name)
            image_path = Path(image_dir) / Path(image_name).name
            if isinstance(ocr_pipeline, dict):
                ocr_output = ocr_pipeline.get(Path(image_path).name, [])
            else:
                ocr_output = ocr_pipeline
            result = self.evaluate_image(image_path, ocr_output)
            results.append(result)
            if not result["parse_success"]:
                failed_parse.append(result["image_path"])
        return {
            "summary": summarize_e2e_results(results),
            "per_image": results,
            "by_format": breakdown_e2e(results, "format"),
            "by_industry": breakdown_e2e(results, "industry"),
            "failed_parse": failed_parse,
        }

    def generate_report(self, results: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
        return write_e2e_outputs(results, output_dir)


def end_to_end_metric_names() -> list[str]:
    return ["cell_accuracy", "worker_schedule_accuracy", "name_accuracy", "code_distribution_error", "parse_success_rate"]


def run_or_wrap_ocr(image_path: str | Path, ocr_pipeline: OCRPipeline | list[dict[str, Any]]) -> list[OCRToken]:
    raw = ocr_pipeline.run(image_path) if hasattr(ocr_pipeline, "run") else ocr_pipeline
    return [
        OCRToken(text=item["text"], bbox=item["bbox"], score=float(item.get("score", 1.0)))
        for item in raw
        if item.get("text") is not None and item.get("bbox") is not None
    ]


def parse_schedule_from_tokens(tokens: list[OCRToken], gt_hint: ScheduleMatrix, config: ParseConfig) -> tuple[ScheduleMatrix, bool]:
    if not tokens:
        return empty_prediction_like(gt_hint), False
    rows = cluster_tokens(tokens, axis="y", ratio=config.row_height_ratio)
    cols = cluster_tokens(tokens, axis="x", ratio=config.col_width_ratio)
    if not rows or not cols:
        return empty_prediction_like(gt_hint), False

    header_row = find_header_row(rows, config.min_header_numeric_ratio)
    name_col_index = 0
    dates = extract_dates(rows[header_row]) or sorted(next(iter(gt_hint.schedule.values())).keys(), key=int)
    schedule: dict[str, dict[str, str]] = {}
    workers: list[str] = []

    for row_index, row_tokens in enumerate(rows):
        if row_index == header_row:
            continue
        sorted_row = sorted(row_tokens, key=lambda token: token.x_center)
        if not sorted_row:
            continue
        worker = sorted_row[name_col_index].text
        if not worker:
            continue
        workers.append(worker)
        schedule[worker] = {day: "" for day in dates}
        for token in sorted_row[name_col_index + 1 :]:
            col_idx = nearest_cluster_index(token, cols, axis="x")
            date_index = max(0, min(len(dates) - 1, col_idx - 1))
            if dates:
                schedule[worker][dates[date_index]] = token.text

    parse_success = bool(workers and dates)
    return (
        ScheduleMatrix(
            image_path=gt_hint.image_path,
            format=gt_hint.format,
            industry=gt_hint.industry,
            year_month=gt_hint.year_month,
            workers=workers,
            schedule=schedule,
            metadata={"parse_success": parse_success},
        ),
        parse_success,
    )


def cluster_tokens(tokens: list[OCRToken], axis: str, ratio: float) -> list[list[OCRToken]]:
    key = (lambda token: token.y_center) if axis == "y" else (lambda token: token.x_center)
    size = (lambda token: token.height) if axis == "y" else (lambda token: token.width)
    sorted_tokens = sorted(tokens, key=key)
    clusters: list[list[OCRToken]] = []
    for token in sorted_tokens:
        if not clusters:
            clusters.append([token])
            continue
        current = clusters[-1]
        current_center = sum(key(item) for item in current) / len(current)
        threshold = (sum(size(item) for item in current) / len(current)) * ratio
        if abs(key(token) - current_center) <= threshold:
            current.append(token)
        else:
            clusters.append([token])
    return [sorted(cluster, key=lambda token: token.x_center) for cluster in clusters]


def find_header_row(rows: list[list[OCRToken]], min_numeric_ratio: float) -> int:
    best_index = 0
    best_ratio = -1.0
    for index, row in enumerate(rows):
        if not row:
            continue
        numeric_count = sum(1 for token in row if token.text.isdigit() and 1 <= int(token.text) <= 31)
        ratio = numeric_count / len(row)
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = index
    return best_index if best_ratio >= min_numeric_ratio else 0


def extract_dates(row: list[OCRToken]) -> list[str]:
    dates = [token.text for token in sorted(row, key=lambda token: token.x_center) if token.text.isdigit() and 1 <= int(token.text) <= 31]
    return dates


def nearest_cluster_index(token: OCRToken, clusters: list[list[OCRToken]], axis: str) -> int:
    value = token.x_center if axis == "x" else token.y_center
    centers = []
    for cluster in clusters:
        if axis == "x":
            centers.append(sum(item.x_center for item in cluster) / len(cluster))
        else:
            centers.append(sum(item.y_center for item in cluster) / len(cluster))
    return min(range(len(centers)), key=lambda index: abs(centers[index] - value))


def empty_prediction_like(gt: ScheduleMatrix) -> ScheduleMatrix:
    return ScheduleMatrix(
        image_path=gt.image_path,
        format=gt.format,
        industry=gt.industry,
        year_month=gt.year_month,
        workers=[],
        schedule={worker: {day: "" for day in days} for worker, days in gt.schedule.items()},
        metadata={"parse_success": False},
    )


def compare_schedule(gt: ScheduleMatrix, pred: ScheduleMatrix, parse_success: bool) -> E2EImageResult:
    total_cells = 0
    correct_cells = 0
    errors: list[dict[str, str]] = []
    for worker, day_map in gt.schedule.items():
        pred_worker = worker if worker in pred.schedule else find_best_worker_match(worker, pred.workers)
        pred_days = pred.schedule.get(pred_worker, {}) if pred_worker else {}
        for day, gt_code in day_map.items():
            total_cells += 1
            pred_code = pred_days.get(day, "")
            if pred_code == gt_code:
                correct_cells += 1
            else:
                errors.append(
                    {
                        "worker": worker,
                        "date": day,
                        "gt_code": gt_code,
                        "pred_code": pred_code,
                        "error_type": classify_e2e_error(gt_code, pred_code, parse_success),
                    }
                )

    worker_matches = sum(worker_schedule_matches(worker, gt.schedule.get(worker, {}), pred.schedule.get(worker, {})) for worker in gt.workers)
    name_matches = sum(1 for worker in gt.workers if worker in pred.workers)
    return E2EImageResult(
        image_path=gt.image_path,
        parse_success=parse_success,
        cell_accuracy=safe_div(correct_cells, total_cells),
        worker_schedule_accuracy=safe_div(worker_matches, len(gt.workers)),
        name_accuracy=safe_div(name_matches, len(gt.workers)),
        code_distribution_error=code_distribution_error(gt.schedule, pred.schedule),
        total_cells=total_cells,
        correct_cells=correct_cells,
        errors=errors,
        format=gt.format,
        industry=gt.industry,
    )


def find_best_worker_match(worker: str, candidates: list[str]) -> str | None:
    return worker if worker in candidates else None


def worker_schedule_matches(worker: str, gt_days: dict[str, str], pred_days: dict[str, str]) -> bool:
    return bool(gt_days) and all(pred_days.get(day, "") == code for day, code in gt_days.items())


def classify_e2e_error(gt_code: str, pred_code: str, parse_success: bool) -> str:
    if not parse_success:
        return "parse_error"
    if pred_code == "":
        return "det_miss"
    if pred_code != gt_code:
        return "rec_error"
    return "unknown"


def code_distribution_error(gt_schedule: dict[str, dict[str, str]], pred_schedule: dict[str, dict[str, str]]) -> float:
    gt_counts = Counter(code for days in gt_schedule.values() for code in days.values() if code)
    pred_counts = Counter(code for days in pred_schedule.values() for code in days.values() if code)
    all_codes = set(gt_counts) | set(pred_counts)
    diff = sum(abs(pred_counts.get(code, 0) - gt_counts.get(code, 0)) for code in all_codes)
    return safe_div(diff, sum(gt_counts.values()))


def summarize_e2e_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result["parse_success"]]
    denominator = len(successful) or len(results)
    summary = {
        "total_images": len(results),
        "parse_success_rate": safe_div(len(successful), len(results)),
        "metrics": {
            "cell_accuracy": safe_div(sum(result["cell_accuracy"] for result in successful), denominator),
            "worker_schedule_accuracy": safe_div(sum(result["worker_schedule_accuracy"] for result in successful), denominator),
            "name_accuracy": safe_div(sum(result["name_accuracy"] for result in successful), denominator),
            "code_distribution_error": safe_div(sum(result["code_distribution_error"] for result in successful), denominator),
        },
        "error_attribution": error_attribution(results),
    }
    summary["targets_met"] = {
        "cell_accuracy": summary["metrics"]["cell_accuracy"] >= 0.90,
        "worker_schedule_accuracy": summary["metrics"]["worker_schedule_accuracy"] >= 0.70,
        "name_accuracy": summary["metrics"]["name_accuracy"] >= 0.85,
        "code_distribution": summary["metrics"]["code_distribution_error"] <= 0.05,
    }
    summary["all_targets_met"] = all(summary["targets_met"].values())
    return summary


def error_attribution(results: list[dict[str, Any]]) -> dict[str, float]:
    counter: Counter[str] = Counter(error["error_type"] for result in results for error in result["errors"])
    total = sum(counter.values())
    return {key: safe_div(value, total) for key, value in sorted(counter.items())}


def breakdown_e2e(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result.get(key, "unknown")].append(result)
    rows: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        summary = summarize_e2e_results(items)
        rows.append(
            {
                "group": group,
                "sample_count": len(items),
                "cell_accuracy": summary["metrics"]["cell_accuracy"],
                "worker_schedule_accuracy": summary["metrics"]["worker_schedule_accuracy"],
                "parse_success_rate": summary["parse_success_rate"],
                "main_error_type": max(summary["error_attribution"], key=summary["error_attribution"].get) if summary["error_attribution"] else "",
            }
        )
    return rows


def write_e2e_outputs(results: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "eval_date": str(date.today()),
        "data_split": "test",
        **results["summary"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    per_image_path = output / "per_image.csv"
    write_csv(per_image_path, flatten_per_image(results["per_image"]))
    error_path = output / "error_attribution.csv"
    write_csv(error_path, [{"error_type": key, "ratio": value} for key, value in results["summary"]["error_attribution"].items()])
    format_path = output / "format_breakdown.csv"
    write_csv(format_path, results["by_format"])
    industry_path = output / "industry_breakdown.csv"
    write_csv(industry_path, results["by_industry"])
    failed_path = output / "failed_parse.txt"
    failed_path.write_text("\n".join(results["failed_parse"]) + ("\n" if results["failed_parse"] else ""), encoding="utf-8")
    report_path = output / "e2e_report.md"
    report_path.write_text(build_e2e_report(results), encoding="utf-8")
    for subdir in ("visualizations/correct_samples", "visualizations/error_samples", "visualizations/schedule_diff"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    return {
        "summary": str(summary_path),
        "per_image": str(per_image_path),
        "error_attribution": str(error_path),
        "format_breakdown": str(format_path),
        "industry_breakdown": str(industry_path),
        "failed_parse": str(failed_path),
        "report": str(report_path),
    }


def flatten_per_image(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "image_path": result["image_path"],
            "parse_success": result["parse_success"],
            "cell_accuracy": result["cell_accuracy"],
            "worker_schedule_accuracy": result["worker_schedule_accuracy"],
            "name_accuracy": result["name_accuracy"],
            "code_distribution_error": result["code_distribution_error"],
            "total_cells": result["total_cells"],
            "correct_cells": result["correct_cells"],
            "format": result["format"],
            "industry": result["industry"],
            "error_count": len(result["errors"]),
        }
        for result in results
    ]


def build_e2e_report(results: dict[str, Any]) -> str:
    summary = results["summary"]
    lines = [
        "# E2E Roster Evaluation Report",
        "",
        "## Summary",
        f"- total_images: {summary['total_images']}",
        f"- parse_success_rate: {summary['parse_success_rate']:.3f}",
        f"- cell_accuracy: {summary['metrics']['cell_accuracy']:.3f}",
        f"- worker_schedule_accuracy: {summary['metrics']['worker_schedule_accuracy']:.3f}",
        f"- name_accuracy: {summary['metrics']['name_accuracy']:.3f}",
        f"- code_distribution_error: {summary['metrics']['code_distribution_error']:.3f}",
        "",
        "## Error Attribution",
        "| error_type | ratio |",
        "|---|---:|",
    ]
    for key, value in summary["error_attribution"].items():
        lines.append(f"| {key} | {value:.1%} |")
    lines.extend(["", "## Failed Parse Images"])
    for image_path in results["failed_parse"]:
        lines.append(f"- {image_path}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
