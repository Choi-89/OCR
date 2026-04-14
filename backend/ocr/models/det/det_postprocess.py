from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DetectionPostProcessConfig:
    prob_threshold: float = 0.3
    box_threshold: float = 0.5
    min_box_size: int = 3
    unclip_ratio: float = 1.5
    max_candidates: int = 1000


def filter_boxes(
    boxes: list[list[int]],
    scores: list[float],
    config: DetectionPostProcessConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or DetectionPostProcessConfig()
    filtered: list[dict[str, Any]] = []
    for box, score in zip(boxes, scores):
        width = max(0, box[2] - box[0])
        height = max(0, box[3] - box[1])
        if score < cfg.box_threshold:
            continue
        if min(width, height) < cfg.min_box_size:
            continue
        filtered.append({"box": box, "score": score})
        if len(filtered) >= cfg.max_candidates:
            break
    return filtered


def unclip_box(box: list[int], ratio: float = 1.5) -> list[int]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    half_w = (box[2] - box[0]) * ratio / 2.0
    half_h = (box[3] - box[1]) * ratio / 2.0
    return [
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
    ]


def reverse_padding(
    box: list[int],
    padding: tuple[int, int, int, int],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> list[int]:
    top, _, left, _ = padding
    return [
        int(round((box[0] - left) / scale_x)),
        int(round((box[1] - top) / scale_y)),
        int(round((box[2] - left) / scale_x)),
        int(round((box[3] - top) / scale_y)),
    ]


def postprocess_boxes(
    boxes: list[list[int]],
    scores: list[float],
    *,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    config: DetectionPostProcessConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or DetectionPostProcessConfig()
    filtered = filter_boxes(boxes, scores, cfg)
    restored: list[dict[str, Any]] = []
    for item in filtered:
        unclipped = unclip_box(item["box"], cfg.unclip_ratio)
        restored.append(
            {
                "box": reverse_padding(unclipped, padding, scale_x=scale_x, scale_y=scale_y),
                "score": item["score"],
            }
        )
    return restored

