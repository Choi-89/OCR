from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ocr_project.stage2_preprocess.preprocess import PreprocessPipeline, ensure_bgr_uint8
from ocr_project.stage6_deployment.export_model import build_predictor


class PredictorLike(Protocol):
    def run(self, inputs: list[np.ndarray]) -> Any:
        ...


@dataclass(slots=True)
class OCRServiceConfig:
    root: Path = field(default_factory=lambda: Path("."))
    enable_ocr: bool = True
    use_gpu: bool = False
    device_id: int = 0
    cpu_threads: int = 4
    preprocess_config: Path = Path("configs/preprocess_config.yaml")
    det_model_dir: Path = Path("backend/ocr/inference/det")
    rec_model_dir: Path = Path("backend/ocr/inference/rec")
    cls_model_dir: Path = Path("backend/ocr/inference/cls")
    dict_path: Path = Path("backend/ocr/inference/rec/dict.txt")


class OCRService:
    """Inference-time OCR pipeline for ShiftFlow.

    The public response contract mirrors the legacy service:
    {"results": [{"text": str, "confidence": float, "bbox": [...] }], "processing_time": float}
    """

    def __init__(
        self,
        config: OCRServiceConfig | None = None,
        *,
        det_predictor: PredictorLike | None = None,
        rec_predictor: PredictorLike | None = None,
        cls_predictor: PredictorLike | None = None,
    ):
        self.config = config or config_from_env()
        self.enabled = self.config.enable_ocr
        self.preprocess = PreprocessPipeline(resolve_path(self.config.root, self.config.preprocess_config))
        self.dictionary = load_dictionary(resolve_path(self.config.root, self.config.dict_path))
        self.det_predictor = det_predictor
        self.rec_predictor = rec_predictor
        self.cls_predictor = cls_predictor
        if self.enabled and not any((det_predictor, rec_predictor, cls_predictor)):
            self._load_models()

    def _load_models(self) -> None:
        self.det_predictor = build_predictor(
            resolve_path(self.config.root, self.config.det_model_dir),
            use_gpu=self.config.use_gpu,
            device_id=self.config.device_id,
            cpu_threads=self.config.cpu_threads,
        )
        self.rec_predictor = build_predictor(
            resolve_path(self.config.root, self.config.rec_model_dir),
            use_gpu=self.config.use_gpu,
            device_id=self.config.device_id,
            cpu_threads=self.config.cpu_threads,
        )
        self.cls_predictor = build_predictor(
            resolve_path(self.config.root, self.config.cls_model_dir),
            use_gpu=self.config.use_gpu,
            device_id=self.config.device_id,
            cpu_threads=self.config.cpu_threads,
        )

    def predict(self, image_input: str | Path | np.ndarray) -> dict[str, Any]:
        start = time.perf_counter()
        if not self.enabled:
            return legacy_response([], time.perf_counter() - start, enabled=False)

        image = load_image(self.preprocess, image_input)
        original_shape = image.shape[:2]
        corrected_image, angle = self._run_angle_classifier(image)
        det_preprocessed = self.preprocess.run(corrected_image, mode="det", angle=0)
        det_input = to_nchw_batch(det_preprocessed["image"])
        raw_boxes, raw_scores = self._run_detection(det_input)
        boxes = self._restore_boxes(raw_boxes, det_preprocessed, original_shape)

        results: list[dict[str, Any]] = []
        for box, det_score in zip(boxes, raw_scores):
            crop = self._crop_image(corrected_image, box)
            if crop.size == 0:
                results.append({"text": "", "confidence": 0.0, "bbox": box, "det_score": float(det_score)})
                continue
            rec_preprocessed = self.preprocess.run(crop, mode="rec", angle=0)
            text, rec_score = self._run_recognition(to_nchw_batch(rec_preprocessed["image"]))
            results.append({"text": text, "confidence": round(float(rec_score), 4), "bbox": box, "det_score": float(det_score)})

        response = legacy_response(results, time.perf_counter() - start, enabled=True)
        response["angle_corrected"] = angle != 0
        response["angle"] = angle
        response["texts"] = [item["text"] for item in results]
        response["boxes"] = [item["bbox"] for item in results]
        response["scores"] = [item["confidence"] for item in results]
        return response

    def _run_angle_classifier(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        if self.cls_predictor is None:
            return image, 0
        raw = self.cls_predictor.run([image])
        angle = parse_angle_output(raw)
        if angle == 180:
            return np.rot90(image, 2).copy(), 180
        return image, 0

    def _run_detection(self, image_batch: np.ndarray) -> tuple[list[list[int]], list[float]]:
        if self.det_predictor is None:
            return [], []
        raw = self.det_predictor.run([image_batch])
        return parse_detection_output(raw)

    def _run_recognition(self, image_batch: np.ndarray) -> tuple[str, float]:
        if self.rec_predictor is None:
            return "", 0.0
        raw = self.rec_predictor.run([image_batch])
        return parse_recognition_output(raw, self.dictionary)

    def _restore_boxes(self, boxes: list[list[int]], preprocess_result: dict[str, Any], original_shape: tuple[int, int]) -> list[list[int]]:
        resized_h, resized_w = preprocess_result["stages"]["resized"].shape[:2]
        original_h, original_w = original_shape
        top, _, left, _ = preprocess_result["padding"]
        scale_x = (resized_w - left) / max(1, original_w)
        scale_y = (resized_h - top) / max(1, original_h)
        restored: list[list[int]] = []
        for box in boxes:
            x1 = int(round((box[0] - left) / max(scale_x, 1e-6)))
            y1 = int(round((box[1] - top) / max(scale_y, 1e-6)))
            x2 = int(round((box[2] - left) / max(scale_x, 1e-6)))
            y2 = int(round((box[3] - top) / max(scale_y, 1e-6)))
            restored.append(clip_box([x1, y1, x2, y2], original_w, original_h))
        return restored

    def _crop_image(self, image: np.ndarray, box: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = clip_box(box, image.shape[1], image.shape[0])
        return image[y1:y2, x1:x2]


_ocr_service_instance: OCRService | None = None


def get_ocr_service(config: OCRServiceConfig | None = None) -> OCRService:
    global _ocr_service_instance
    if _ocr_service_instance is None:
        _ocr_service_instance = OCRService(config)
    return _ocr_service_instance


def reset_ocr_service() -> None:
    global _ocr_service_instance
    _ocr_service_instance = None


def config_from_env(root: str | Path = ".") -> OCRServiceConfig:
    base = Path(root)
    return OCRServiceConfig(
        root=base,
        enable_ocr=parse_bool(os.getenv("ENABLE_OCR", "true")),
        use_gpu=parse_bool(os.getenv("ENABLE_GPU", "false")),
        device_id=int(os.getenv("OCR_GPU_DEVICE_ID", "0")),
        cpu_threads=int(os.getenv("OCR_CPU_THREADS", "4")),
    )


def load_image(preprocess: PreprocessPipeline, image_input: str | Path | np.ndarray) -> np.ndarray:
    loaded = preprocess._load_and_normalize(image_input)
    return loaded["image"]


def legacy_response(results: list[dict[str, Any]], processing_time: float, *, enabled: bool) -> dict[str, Any]:
    return {"results": results, "processing_time": round(processing_time, 4), "enabled": enabled}


def to_nchw_batch(image: np.ndarray) -> np.ndarray:
    array = image.astype("float32", copy=False)
    if array.ndim == 3 and array.shape[-1] == 3:
        array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def parse_angle_output(raw: Any) -> int:
    if isinstance(raw, dict):
        if "angle" in raw:
            return int(raw["angle"])
        if "label" in raw:
            return 180 if str(raw["label"]) in {"1", "180"} else 0
        raw = raw.get("probabilities", raw)
    values = np.asarray(raw)
    if values.size >= 2:
        flat = values.reshape(-1)
        return 180 if int(np.argmax(flat[-2:])) == 1 else 0
    return 0


def parse_detection_output(raw: Any) -> tuple[list[list[int]], list[float]]:
    if isinstance(raw, dict):
        boxes = raw.get("boxes", [])
        scores = raw.get("scores", [1.0] * len(boxes))
        return [[int(v) for v in box] for box in boxes], [float(score) for score in scores]
    if isinstance(raw, tuple) and len(raw) == 2:
        boxes, scores = raw
        return [[int(v) for v in box] for box in boxes], [float(score) for score in scores]
    return [], []


def parse_recognition_output(raw: Any, dictionary: list[str]) -> tuple[str, float]:
    if isinstance(raw, dict):
        if "text" in raw:
            return str(raw["text"]), float(raw.get("score", raw.get("confidence", 0.0)))
        texts = raw.get("texts")
        scores = raw.get("scores")
        if texts:
            return str(texts[0]), float(scores[0] if scores else 0.0)
    return "", 0.0


def load_dictionary(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.exists():
        return []
    return source.read_text(encoding="utf-8").splitlines()


def clip_box(box: list[int], width: int, height: int) -> list[int]:
    x1 = max(0, min(width, int(box[0])))
    y1 = max(0, min(height, int(box[1])))
    x2 = max(0, min(width, int(box[2])))
    y2 = max(0, min(height, int(box[3])))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in {"0", "false", "no", "off"}


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def make_blank_image(width: int = 200, height: int = 200) -> np.ndarray:
    return ensure_bgr_uint8(np.full((height, width, 3), 255, dtype=np.uint8))
