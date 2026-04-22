from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_preview(image: np.ndarray, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    cv2.imwrite(str(target), (np.clip(image, 0, 1) * 255).astype(np.uint8))
    return target


def save_curve_png(
    output_path: str | Path,
    x_values: list[float],
    y_values: list[float],
    *,
    title: str,
    ylabel: str,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(x_values, y_values)
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(target)
        plt.close()
    except Exception:
        target.write_text("matplotlib_unavailable\n", encoding="utf-8")
    return target


def build_recognition_error_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "image": row.get("image", ""),
            "target": row.get("target", ""),
            "prediction": row.get("prediction", ""),
            "match": str(row.get("target", "") == row.get("prediction", "")),
            "cer": row.get("cer", ""),
        }
        for row in rows
    ]
