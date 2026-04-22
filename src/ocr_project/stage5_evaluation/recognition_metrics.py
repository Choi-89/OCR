from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


TEXT_TYPES: tuple[str, ...] = ("single_char", "date", "handwrite", "normal")
WORK_CODES: set[str] = {
    "D",
    "E",
    "N",
    "A",
    "B",
    "C",
    "OFF",
    "OT",
    "AL",
    "SL",
    "CL",
    "PH",
    "AM",
    "PM",
    "OP",
    "MD",
    "FT",
    "PT",
    "Open",
    "Close",
    "Mid",
    "Full",
    "Part",
}


@dataclass(slots=True)
class EditOperation:
    op: str
    gt_char: str
    pred_char: str


@dataclass(slots=True)
class RecSampleRecord:
    image_path: str
    gt_text: str
    pred_text: str
    text_type: str
    edit_distance: int
    cer: float
    normalized_cer: float
    is_correct: bool
    operations: list[EditOperation] = field(default_factory=list)


@dataclass(slots=True)
class RecognitionEvaluator:
    dict_path: str | Path | None = None
    ignore_case: bool = False
    records: list[RecSampleRecord] = field(default_factory=list)

    def update(
        self,
        pred_texts: list[str],
        gt_texts: list[str],
        text_types: list[str] | None = None,
        image_paths: list[str] | None = None,
    ) -> None:
        text_types = text_types or [infer_text_type(gt, path) for gt, path in zip(gt_texts, image_paths or [])]
        image_paths = image_paths or [f"crop_{len(self.records) + index}.png" for index in range(len(pred_texts))]
        for pred, gt, text_type, image_path in zip(pred_texts, gt_texts, text_types, image_paths):
            if gt == "###" or gt == "":
                continue
            pred_eval = pred.lower() if self.ignore_case else pred
            gt_eval = gt.lower() if self.ignore_case else gt
            distance, operations = levenshtein_with_ops(pred_eval, gt_eval)
            if len(pred_eval) > max(1, len(gt_eval) * 5):
                distance = min(distance, max(len(pred_eval), len(gt_eval)))
            cer = safe_div(distance, len(gt_eval))
            normalized_cer = safe_div(distance, max(len(gt_eval), len(pred_eval), 1))
            self.records.append(
                RecSampleRecord(
                    image_path=image_path,
                    gt_text=gt,
                    pred_text=pred,
                    text_type=text_type if text_type in TEXT_TYPES else infer_text_type(gt, image_path),
                    edit_distance=distance,
                    cer=cer,
                    normalized_cer=normalized_cer,
                    is_correct=pred_eval == gt_eval,
                    operations=operations,
                )
            )

    def compute(self) -> dict[str, Any]:
        total_edit = sum(record.edit_distance for record in self.records)
        total_chars = sum(len(record.gt_text) for record in self.records)
        total_samples = len(self.records)
        correct = sum(1 for record in self.records if record.is_correct)
        metrics = {
            "cer": safe_div(total_edit, total_chars),
            "wer": safe_div(total_samples - correct, total_samples),
            "accuracy": safe_div(correct, total_samples),
            "normalized_cer": safe_div(sum(record.normalized_cer for record in self.records), total_samples),
            "total_chars": total_chars,
            "total_edit_distance": total_edit,
            "total_samples": total_samples,
            "correct_samples": correct,
            "by_type": self.compute_by_type(),
            "domain_metrics": self.compute_domain_metrics(),
        }
        metrics["targets_met"] = {
            "cer": metrics["cer"] <= 0.03,
            "wer": metrics["wer"] <= 0.05,
            "accuracy": metrics["accuracy"] >= 0.90,
            "work_code": metrics["domain_metrics"]["work_code_accuracy"] >= 0.95,
        }
        metrics["all_targets_met"] = all(metrics["targets_met"].values())
        return metrics

    def compute_by_type(self) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for text_type in TEXT_TYPES:
            subset = [record for record in self.records if record.text_type == text_type]
            total_edit = sum(record.edit_distance for record in subset)
            total_chars = sum(len(record.gt_text) for record in subset)
            correct = sum(1 for record in subset if record.is_correct)
            output[text_type] = {
                "cer": safe_div(total_edit, total_chars),
                "accuracy": safe_div(correct, len(subset)),
                "count": len(subset),
            }
        return output

    def compute_domain_metrics(self) -> dict[str, float]:
        work_code_records = [record for record in self.records if record.gt_text in WORK_CODES]
        date_records = [record for record in self.records if record.text_type == "date"]
        return {
            "work_code_accuracy": safe_div(sum(record.is_correct for record in work_code_records), len(work_code_records)),
            "date_exact_match": safe_div(sum(record.is_correct for record in date_records), len(date_records)),
        }

    def compute_per_sample(self) -> list[dict[str, Any]]:
        return [
            {
                "image_path": record.image_path,
                "gt_text": record.gt_text,
                "pred_text": record.pred_text,
                "text_type": record.text_type,
                "edit_distance": record.edit_distance,
                "cer": record.cer,
                "normalized_cer": record.normalized_cer,
                "is_correct": record.is_correct,
            }
            for record in self.records
        ]

    def confusion_pairs(self, top_k: int = 20) -> list[dict[str, Any]]:
        counter: Counter[tuple[str, str]] = Counter()
        for record in self.records:
            for op in record.operations:
                if op.op == "substitute":
                    counter[(op.gt_char, op.pred_char)] += 1
        return [
            {"gt_char": gt, "pred_char": pred, "count": count}
            for (gt, pred), count in counter.most_common(top_k)
        ]

    def length_breakdown(self) -> list[dict[str, Any]]:
        buckets: dict[str, list[RecSampleRecord]] = defaultdict(list)
        for record in self.records:
            buckets[length_bucket(len(record.gt_text))].append(record)
        rows: list[dict[str, Any]] = []
        for bucket in ("1", "2", "3-5", "6-10", "11+"):
            subset = buckets[bucket]
            rows.append(
                {
                    "length_bucket": bucket,
                    "sample_count": len(subset),
                    "avg_cer": safe_div(sum(record.cer for record in subset), len(subset)),
                }
            )
        return rows

    def reset(self) -> None:
        self.records.clear()


def recognition_metric_names() -> list[str]:
    return ["cer", "wer", "accuracy", "normalized_cer", "work_code_accuracy", "date_exact_match"]


def levenshtein(pred: str, gt: str) -> int:
    return levenshtein_with_ops(pred, gt)[0]


def levenshtein_with_ops(pred: str, gt: str) -> tuple[int, list[EditOperation]]:
    pred_chars = list(pred)
    gt_chars = list(gt)
    rows = len(pred_chars) + 1
    cols = len(gt_chars) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(cols):
        dp[0][j] = j
        back[0][j] = "insert"
    back[0][0] = "match"

    for i in range(1, rows):
        for j in range(1, cols):
            if pred_chars[i - 1] == gt_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "match"
            else:
                options = [
                    (dp[i - 1][j] + 1, "delete"),
                    (dp[i][j - 1] + 1, "insert"),
                    (dp[i - 1][j - 1] + 1, "substitute"),
                ]
                dp[i][j], back[i][j] = min(options, key=lambda item: item[0])

    operations: list[EditOperation] = []
    i = len(pred_chars)
    j = len(gt_chars)
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "match":
            i -= 1
            j -= 1
        elif op == "substitute":
            operations.append(EditOperation("substitute", gt_chars[j - 1], pred_chars[i - 1]))
            i -= 1
            j -= 1
        elif op == "delete":
            operations.append(EditOperation("delete", "", pred_chars[i - 1]))
            i -= 1
        else:
            operations.append(EditOperation("insert", gt_chars[j - 1], ""))
            j -= 1
    operations.reverse()
    return dp[-1][-1], operations


def infer_text_type(gt_text: str, image_path: str = "") -> str:
    if "handwrite" in image_path.lower():
        return "handwrite"
    if len(gt_text) == 1:
        return "single_char"
    if re.fullmatch(r"[0-9./:\-~ ]+", gt_text):
        return "date"
    return "normal"


def length_bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 5:
        return "3-5"
    if length <= 10:
        return "6-10"
    return "11+"


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def write_recognition_eval_outputs(
    evaluator: RecognitionEvaluator,
    output_dir: str | Path,
    *,
    checkpoint: str,
    data_split: str,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluator.compute()
    summary = {
        "eval_date": str(date.today()),
        "checkpoint": checkpoint,
        "data_split": data_split,
        "metrics": {
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "accuracy": metrics["accuracy"],
            "normalized_cer": metrics["normalized_cer"],
        },
        "by_type": metrics["by_type"],
        "domain_metrics": metrics["domain_metrics"],
        "targets_met": metrics["targets_met"],
        "all_targets_met": metrics["all_targets_met"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    per_sample_path = output / "per_sample.csv"
    write_csv(per_sample_path, evaluator.compute_per_sample())
    confusion_path = output / "confusion_pairs.csv"
    write_csv(confusion_path, evaluator.confusion_pairs())
    type_path = output / "type_breakdown.csv"
    write_csv(type_path, [{"text_type": key, **value} for key, value in metrics["by_type"].items()])
    length_path = output / "length_breakdown.csv"
    write_csv(length_path, evaluator.length_breakdown())
    error_path = output / "error_analysis.md"
    error_path.write_text(build_recognition_error_report(evaluator), encoding="utf-8")
    for subdir in ("visualizations/high_cer_samples", "visualizations/type_samples"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    return {
        "summary": str(summary_path),
        "per_sample": str(per_sample_path),
        "confusion_pairs": str(confusion_path),
        "type_breakdown": str(type_path),
        "length_breakdown": str(length_path),
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


def build_recognition_error_report(evaluator: RecognitionEvaluator) -> str:
    metrics = evaluator.compute()
    high_cer = sorted(evaluator.records, key=lambda record: record.cer, reverse=True)[:50]
    lines = [
        "# Recognition Error Analysis Report",
        "",
        "## Summary",
        f"- CER: {metrics['cer']:.4f}",
        f"- WER: {metrics['wer']:.4f}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        "",
        "## Type Breakdown",
        "| type | count | CER | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for text_type, values in metrics["by_type"].items():
        lines.append(f"| {text_type} | {values['count']} | {values['cer']:.4f} | {values['accuracy']:.4f} |")
    lines.extend(["", "## Top Confusion Pairs", "| gt | pred | count |", "|---|---|---:|"])
    for row in evaluator.confusion_pairs(20):
        lines.append(f"| {row['gt_char']} | {row['pred_char']} | {row['count']} |")
    lines.extend(["", "## High CER Samples Top 50"])
    for index, record in enumerate(high_cer, start=1):
        lines.append(f"{index}. {record.image_path} | gt={record.gt_text} | pred={record.pred_text} | CER={record.cer:.3f}")
    return "\n".join(lines) + "\n"


def evaluate_recognition(
    pred_texts: list[str],
    gt_texts: list[str],
    text_types: list[str] | None = None,
    image_paths: list[str] | None = None,
    *,
    dict_path: str | Path | None = None,
    ignore_case: bool = False,
) -> dict[str, Any]:
    evaluator = RecognitionEvaluator(dict_path=dict_path, ignore_case=ignore_case)
    evaluator.update(pred_texts, gt_texts, text_types, image_paths)
    return evaluator.compute()
