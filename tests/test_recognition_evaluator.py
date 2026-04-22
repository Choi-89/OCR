from __future__ import annotations

from ocr_project.stage5_evaluation.recognition_metrics import (
    RecognitionEvaluator,
    infer_text_type,
    levenshtein,
)


def test_levenshtein_exact_match_is_zero() -> None:
    assert levenshtein("OFF", "OFF") == 0


def test_levenshtein_full_mismatch_same_length() -> None:
    assert levenshtein("D", "O") == 1


def test_levenshtein_counts_hangul_as_single_character() -> None:
    assert levenshtein("\ub0e5", "\ub0ae") == 1


def test_recognition_evaluator_skips_unreadable_label() -> None:
    evaluator = RecognitionEvaluator()
    evaluator.update(["ABC"], ["###"], image_paths=["crop/a.png"])
    assert evaluator.compute()["total_samples"] == 0


def test_empty_prediction_counts_as_deletions() -> None:
    evaluator = RecognitionEvaluator()
    evaluator.update([""], ["OFF"], image_paths=["crop/a.png"])
    metrics = evaluator.compute()
    assert metrics["total_edit_distance"] == 3
    assert metrics["cer"] == 1.0


def test_text_type_inference() -> None:
    assert infer_text_type("D", "crop/a.png") == "single_char"
    assert infer_text_type("09:00", "crop/a.png") == "date"
    assert infer_text_type("\uadfc\ubb34", "handwrite_sample.png") == "handwrite"


def test_recognition_evaluator_metrics_and_domain_accuracy() -> None:
    evaluator = RecognitionEvaluator()
    evaluator.update(
        ["D", "OEF", "09:00"],
        ["D", "OFF", "09:00"],
        image_paths=["crop/a.png", "crop/b.png", "crop/c.png"],
    )
    metrics = evaluator.compute()
    assert metrics["total_samples"] == 3
    assert metrics["correct_samples"] == 2
    assert round(metrics["accuracy"], 3) == 0.667
    assert metrics["domain_metrics"]["work_code_accuracy"] == 0.5
    assert metrics["domain_metrics"]["date_exact_match"] == 1.0
