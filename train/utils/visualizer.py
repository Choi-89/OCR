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
