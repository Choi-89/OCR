from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


DICT_VERSION = "1.0.0"
SPECIAL_TOKENS: tuple[str, ...] = ("<unk>", "<pad>", "<blank>")
HANGUL_JAMO: tuple[str, ...] = tuple("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㄲㄸㅃㅆㅉ")
UPPERCASE: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWERCASE: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")
DIGITS: tuple[str, ...] = tuple("0123456789")
PUNCTUATION: tuple[str, ...] = (
    ".",
    ":",
    "/",
    "-",
    "(",
    ")",
    "[",
    "]",
    "~",
    "&",
    "+",
    "*",
    "#",
    "@",
    ",",
    "·",
    "%",
    "'",
    '"',
    "\\",
    " ",
)
DOMAIN_SYMBOL_CANDIDATES: tuple[str, ...] = ("△", "○", "●", "×", "✓", "→")

DOMAIN_CODES: dict[str, tuple[str, ...]] = {
    "hospital": (
        "낮",
        "밤",
        "저녁",
        "오전",
        "오후",
        "야간",
        "주간",
        "당직",
        "오프",
        "휴무",
        "반차",
        "연차",
        "대휴",
        "보상",
        "출장",
        "재택",
        "교육",
        "정상",
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
        "D당직",
        "E당직",
        "N당직",
    ),
    "convenience": (
        "조",
        "중",
        "석",
        "야",
        "오픈",
        "마감",
        "미들",
        "클로징",
        "풀타임",
        "파트",
        "OP",
        "CL",
        "MD",
        "FT",
        "PT",
        "06",
        "09",
        "12",
        "15",
        "18",
        "21",
        "24",
    ),
    "factory": (
        "주간",
        "야간",
        "교대",
        "비번",
        "휴무",
        "특근",
        "잔업",
        "조출",
        "A조",
        "B조",
        "C조",
        "D조",
        "1조",
        "2조",
        "3조",
        "4조",
    ),
    "office": (
        "정상",
        "재택",
        "반차오전",
        "반차오후",
        "연차",
        "출장",
        "외근",
        "교육",
        "휴직",
        "재",
        "반",
        "연",
        "출",
        "외",
    ),
    "food": (
        "Open",
        "Close",
        "Mid",
        "Full",
        "Part",
    ),
}


@dataclass(slots=True)
class DictionaryPaths:
    dict_dir: Path
    versioned_dict: Path
    latest_dict: Path
    meta_json: Path
    freq_txt: Path
    chars_from_annotation: Path


@dataclass(slots=True)
class DictionaryBuildResult:
    vocab: list[str]
    frequencies: dict[str, int]
    paths: DictionaryPaths
    meta: dict[str, Any]


def build_payroll_charset() -> list[str]:
    """Build the full ShiftFlow OCR vocabulary excluding special tokens."""
    annotation_chars = set()
    domain_chars = extract_chars_from_strings(
        token
        for industry_tokens in DOMAIN_CODES.values()
        for token in industry_tokens
    )
    base_chars = set(iter_hangul_complete()) | set(HANGUL_JAMO) | set(UPPERCASE) | set(LOWERCASE) | set(DIGITS) | set(PUNCTUATION)
    return sorted(base_chars | domain_chars | annotation_chars)


def iter_hangul_complete() -> list[str]:
    return [chr(codepoint) for codepoint in range(0xAC00, 0xD7A4)]


def extract_annotation_characters(rec_gt_path: str | Path) -> Counter[str]:
    """Read PaddleOCR rec_gt.txt and count per-character frequency."""
    counter: Counter[str] = Counter()
    path = Path(rec_gt_path)
    if not path.exists():
        return counter

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if not line or " " not in line:
            continue
        _, text = line.split(" ", 1)
        counter.update(text)
    return counter


def extract_chars_from_strings(strings: Any) -> set[str]:
    chars: set[str] = set()
    for token in strings:
        chars.update(token)
    return chars


def collect_domain_symbols(annotation_counter: Counter[str]) -> list[str]:
    return [symbol for symbol in DOMAIN_SYMBOL_CANDIDATES if annotation_counter.get(symbol, 0) > 0]


def build_dictionary(
    rec_gt_path: str | Path,
    output_dir: str | Path,
    *,
    version: str = DICT_VERSION,
    include_attention_tokens: bool = False,
) -> DictionaryBuildResult:
    output_root = Path(output_dir)
    paths = initialize_dictionary_paths(output_root, version)
    annotation_counter = extract_annotation_characters(rec_gt_path)
    annotation_chars = set(annotation_counter)
    domain_chars = extract_chars_from_strings(
        token
        for industry_tokens in DOMAIN_CODES.values()
        for token in industry_tokens
    )
    domain_symbols = set(collect_domain_symbols(annotation_counter))

    vocab = sorted(
        set(iter_hangul_complete())
        | set(HANGUL_JAMO)
        | set(UPPERCASE)
        | set(LOWERCASE)
        | set(DIGITS)
        | set(PUNCTUATION)
        | domain_chars
        | annotation_chars
        | domain_symbols
    )

    special_tokens = list(SPECIAL_TOKENS)
    if include_attention_tokens:
        special_tokens = ["<bos>", "<eos>", *special_tokens]
    vocab_with_tokens = [*vocab, *special_tokens]

    paths.versioned_dict.write_text("\n".join(vocab_with_tokens) + "\n", encoding="utf-8")
    update_latest_dict(paths.versioned_dict, paths.latest_dict)
    write_annotation_chars(paths.chars_from_annotation, annotation_counter)
    write_frequency_file(paths.freq_txt, vocab, annotation_counter)

    meta = build_meta(vocab, special_tokens, annotation_counter, domain_symbols, version)
    paths.meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return DictionaryBuildResult(vocab=vocab_with_tokens, frequencies=dict(annotation_counter), paths=paths, meta=meta)


def initialize_dictionary_paths(output_dir: Path, version: str) -> DictionaryPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    return DictionaryPaths(
        dict_dir=output_dir,
        versioned_dict=output_dir / f"dict_v{version}.txt",
        latest_dict=output_dir / "dict_latest.txt",
        meta_json=output_dir / "dict_meta.json",
        freq_txt=output_dir / "dict_freq.txt",
        chars_from_annotation=output_dir / "chars_from_annotation.txt",
    )


def update_latest_dict(versioned_dict: Path, latest_dict: Path) -> None:
    if latest_dict.exists() or latest_dict.is_symlink():
        latest_dict.unlink()
    try:
        latest_dict.symlink_to(versioned_dict.name)
    except OSError:
        latest_dict.write_text(versioned_dict.read_text(encoding="utf-8"), encoding="utf-8")


def write_annotation_chars(output_path: Path, annotation_counter: Counter[str]) -> None:
    lines = [f"{char}\t{count}" for char, count in sorted(annotation_counter.items(), key=lambda item: (-item[1], item[0]))]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_frequency_file(output_path: Path, vocab: list[str], annotation_counter: Counter[str]) -> None:
    annotation_chars = set(annotation_counter)
    domain_chars = extract_chars_from_strings(
        token
        for industry_tokens in DOMAIN_CODES.values()
        for token in industry_tokens
    )
    rows: list[str] = []
    for char in vocab:
        if char in annotation_chars:
            source = "annotation"
        elif char in domain_chars:
            source = "domain_knowledge"
        else:
            source = "base_charset"
        rows.append(f"{char}\t{annotation_counter.get(char, 0)}\t{source}")
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def build_meta(
    vocab: list[str],
    special_tokens: list[str],
    annotation_counter: Counter[str],
    domain_symbols: set[str],
    version: str,
) -> dict[str, Any]:
    special_indices = {token.strip("<>"): len(vocab) + index for index, token in enumerate(special_tokens)}
    return {
        "version": version,
        "created_date": str(date.today()),
        "total_chars": len(vocab) + len(special_tokens),
        "breakdown": {
            "hangul_complete": len(iter_hangul_complete()),
            "hangul_jamo": len(HANGUL_JAMO),
            "uppercase": len(UPPERCASE),
            "lowercase": len(LOWERCASE),
            "digits": len(DIGITS),
            "punctuation": len(PUNCTUATION),
            "domain_symbols": len(domain_symbols),
            "special_tokens": len(special_tokens),
        },
        "special_token_indices": special_indices,
        "sources": ["annotation_rec_gt", "domain_knowledge", "base_charset"],
        "notes": f"annotation_unique_chars={len(annotation_counter)}",
    }


def validate_dictionary(dict_path: str | Path, rec_gt_path: str | Path) -> list[str]:
    dict_chars = load_dictionary(dict_path)
    annotation_counter = extract_annotation_characters(rec_gt_path)
    issues: list[str] = []

    missing = sorted(set(annotation_counter) - set(dict_chars))
    if missing:
        issues.append(f"missing_annotation_chars: {''.join(missing)}")
    duplicates = find_duplicates(dict_chars)
    if duplicates:
        issues.append(f"duplicate_entries: {','.join(duplicates[:20])}")
    if not dict_chars or dict_chars[-1] != "<blank>":
        issues.append("blank_token_not_last")
    try:
        Path(dict_path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        issues.append("dict_not_utf8")
    return issues


def load_dictionary(dict_path: str | Path) -> list[str]:
    return Path(dict_path).read_text(encoding="utf-8").splitlines()


def find_duplicates(chars: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for char in chars:
        if char in seen and char not in duplicates:
            duplicates.append(char)
        seen.add(char)
    return duplicates


def estimate_vocab_size(include_attention_tokens: bool = False) -> int:
    size = len(build_payroll_charset()) + len(SPECIAL_TOKENS)
    if include_attention_tokens:
        size += 2
    return size
