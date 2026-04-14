from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CTCDecoderConfig:
    method: str = "greedy"
    beam_width: int = 5
    blank_index: int = 11314


def greedy_decode(
    logits: np.ndarray,
    dictionary: list[str],
    blank_index: int,
) -> dict[str, list[Any]]:
    indices = np.argmax(logits, axis=-1)
    probabilities = softmax(logits, axis=-1)
    texts: list[str] = []
    scores: list[float] = []
    for batch_index in range(indices.shape[0]):
        collapsed: list[int] = []
        confidences: list[float] = []
        previous = None
        for time_index, char_index in enumerate(indices[batch_index]):
            idx = int(char_index)
            if idx == blank_index:
                previous = idx
                continue
            if previous == idx:
                continue
            collapsed.append(idx)
            confidences.append(float(np.max(probabilities[batch_index, time_index])))
            previous = idx
        texts.append("".join(dictionary[idx] for idx in collapsed if idx < len(dictionary)))
        scores.append(float(np.mean(confidences)) if confidences else 0.0)
    return {"texts": texts, "scores": scores}


def beam_search_decode(
    logits: np.ndarray,
    dictionary: list[str],
    blank_index: int,
    beam_width: int = 5,
) -> dict[str, list[Any]]:
    probabilities = softmax(logits, axis=-1)
    texts: list[str] = []
    scores: list[float] = []
    for batch_index in range(probabilities.shape[0]):
        beams: list[tuple[str, float, int | None]] = [("", 1.0, None)]
        for timestep in probabilities[batch_index]:
            top_indices = np.argsort(timestep)[-beam_width:][::-1]
            next_beams: list[tuple[str, float, int | None]] = []
            for prefix, prefix_score, prev_idx in beams:
                for idx_raw in top_indices:
                    idx = int(idx_raw)
                    next_score = prefix_score * float(timestep[idx])
                    if idx == blank_index:
                        next_beams.append((prefix, next_score, idx))
                    else:
                        char = dictionary[idx] if idx < len(dictionary) else ""
                        next_prefix = prefix if prev_idx == idx else prefix + char
                        next_beams.append((next_prefix, next_score, idx))
            next_beams.sort(key=lambda item: item[1], reverse=True)
            beams = next_beams[:beam_width]
        best_text, best_score, _ = beams[0]
        texts.append(best_text)
        scores.append(best_score)
    return {"texts": texts, "scores": scores}


def decode_logits(
    logits: np.ndarray,
    dictionary: list[str],
    config: CTCDecoderConfig | None = None,
    *,
    text_type: str | None = None,
    max_output_chars: int = 1,
) -> dict[str, list[Any]]:
    cfg = config or CTCDecoderConfig()
    if cfg.method == "beam_search":
        result = beam_search_decode(logits, dictionary, cfg.blank_index, cfg.beam_width)
    else:
        result = greedy_decode(logits, dictionary, cfg.blank_index)
    if text_type == "single_char":
        result["texts"] = [text[:max_output_chars] for text in result["texts"]]
    return result


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
