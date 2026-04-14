from __future__ import annotations

from pathlib import Path

from ocr_project.stage2_preprocess.korean_charset import (
    build_dictionary,
    estimate_vocab_size,
    extract_annotation_characters,
    find_duplicates,
    validate_dictionary,
)


def test_extract_annotation_characters_counts_chars(tmp_path: Path) -> None:
    rec_gt = tmp_path / "rec_gt.txt"
    rec_gt.write_text("crop/a.png D\ncrop/b.png 낮\n", encoding="utf-8")
    counter = extract_annotation_characters(rec_gt)
    assert counter["D"] == 1
    assert counter["낮"] == 1


def test_build_dictionary_writes_expected_outputs(tmp_path: Path) -> None:
    rec_gt = tmp_path / "rec_gt.txt"
    rec_gt.write_text("crop/a.png D\ncrop/b.png 낮\n", encoding="utf-8")
    result = build_dictionary(rec_gt, tmp_path / "dict")
    assert result.paths.versioned_dict.exists()
    assert result.paths.meta_json.exists()
    assert result.paths.freq_txt.exists()
    assert result.paths.chars_from_annotation.exists()


def test_validate_dictionary_requires_blank_last(tmp_path: Path) -> None:
    rec_gt = tmp_path / "rec_gt.txt"
    rec_gt.write_text("crop/a.png D\n", encoding="utf-8")
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("D\n<blank>\n", encoding="utf-8")
    issues = validate_dictionary(dict_path, rec_gt)
    assert "blank_token_not_last" not in issues


def test_find_duplicates_returns_duplicate_chars() -> None:
    assert find_duplicates(["가", "나", "가"]) == ["가"]


def test_estimate_vocab_size_is_large_enough() -> None:
    assert estimate_vocab_size() > 11000
