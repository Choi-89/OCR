from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(slots=True)
class ToggleConfig:
    enabled: bool = True
    prob: float = 0.5


@dataclass(slots=True)
class RangeConfig:
    enabled: bool = True
    prob: float = 0.5
    range: tuple[float, float] = (0.0, 1.0)


@dataclass(slots=True)
class CropConfig:
    enabled: bool = True
    prob: float = 0.6
    min_area_ratio: float = 0.75
    bbox_overlap_thresh: float = 0.5


@dataclass(slots=True)
class PerspectiveConfig:
    enabled: bool = True
    prob: float = 0.3
    distort_limit: float = 0.05


@dataclass(slots=True)
class SigmaConfig:
    enabled: bool = True
    prob: float = 0.4
    sigma_range: tuple[float, float] = (1.0, 8.0)


@dataclass(slots=True)
class KernelConfig:
    enabled: bool = True
    prob: float = 0.2
    kernel_range: tuple[int, int] = (3, 5)


@dataclass(slots=True)
class QualityConfig:
    enabled: bool = True
    prob: float = 0.5
    quality_range: tuple[int, int] = (50, 95)


@dataclass(slots=True)
class DetectionGeometricConfig:
    rotate: RangeConfig = field(default_factory=lambda: RangeConfig(range=(-10.0, 10.0)))
    random_crop: CropConfig = field(default_factory=CropConfig)
    horizontal_flip: ToggleConfig = field(default_factory=lambda: ToggleConfig(prob=0.3))
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    resize_jitter: RangeConfig = field(default_factory=lambda: RangeConfig(range=(0.5, 2.0)))


@dataclass(slots=True)
class DetectionPhotometricConfig:
    brightness: RangeConfig = field(default_factory=lambda: RangeConfig(prob=0.7, range=(0.6, 1.4)))
    contrast: RangeConfig = field(default_factory=lambda: RangeConfig(prob=0.6, range=(0.7, 1.5)))
    hsv_shift: ToggleConfig = field(default_factory=lambda: ToggleConfig(prob=0.4))
    gaussian_noise: SigmaConfig = field(default_factory=lambda: SigmaConfig(prob=0.4, sigma_range=(1.0, 8.0)))
    motion_blur: KernelConfig = field(default_factory=lambda: KernelConfig(prob=0.2, kernel_range=(3, 5)))
    jpeg_quality: QualityConfig = field(default_factory=lambda: QualityConfig(prob=0.5, quality_range=(50, 95)))
    shadow: ToggleConfig = field(default_factory=lambda: ToggleConfig(prob=0.25))


@dataclass(slots=True)
class RecognitionGeometricConfig:
    rotate: RangeConfig = field(default_factory=lambda: RangeConfig(prob=0.3, range=(-3.0, 3.0)))
    stretch_x: RangeConfig = field(default_factory=lambda: RangeConfig(prob=0.3, range=(0.9, 1.1)))
    stretch_y: RangeConfig = field(default_factory=lambda: RangeConfig(prob=0.2, range=(0.9, 1.05)))


@dataclass(slots=True)
class RecognitionPhotometricConfig:
    brightness: RangeConfig = field(default_factory=lambda: RangeConfig(prob=0.7, range=(0.5, 1.5)))
    contrast: RangeConfig = field(default_factory=lambda: RangeConfig(prob=0.6, range=(0.6, 1.6)))
    gaussian_noise: SigmaConfig = field(default_factory=lambda: SigmaConfig(prob=0.5, sigma_range=(1.0, 5.0)))
    gaussian_blur: SigmaConfig = field(default_factory=lambda: SigmaConfig(prob=0.3, sigma_range=(0.3, 1.2)))
    jpeg_quality: QualityConfig = field(default_factory=lambda: QualityConfig(prob=0.6, quality_range=(40, 90)))
    dilate: KernelConfig = field(default_factory=lambda: KernelConfig(prob=0.25, kernel_range=(1, 2)))
    erode: KernelConfig = field(default_factory=lambda: KernelConfig(prob=0.25, kernel_range=(1, 2)))
    grayscale: ToggleConfig = field(default_factory=lambda: ToggleConfig(prob=0.2))


@dataclass(slots=True)
class DetectionAugConfig:
    geometric: DetectionGeometricConfig = field(default_factory=DetectionGeometricConfig)
    photometric: DetectionPhotometricConfig = field(default_factory=DetectionPhotometricConfig)


@dataclass(slots=True)
class RecognitionAugConfig:
    geometric: RecognitionGeometricConfig = field(default_factory=RecognitionGeometricConfig)
    photometric: RecognitionPhotometricConfig = field(default_factory=RecognitionPhotometricConfig)


@dataclass(slots=True)
class AugmentationConfig:
    enabled: bool = True
    seed: int = 42
    det: DetectionAugConfig = field(default_factory=DetectionAugConfig)
    rec: RecognitionAugConfig = field(default_factory=RecognitionAugConfig)


class AugmentPipeline:
    """Training-only online augmentation pipeline."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = load_augment_config(self.config_path)

    def run_det(self, image: np.ndarray, bboxes: list[list[int]], seed: int | None = None) -> dict[str, Any]:
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        frame = ensure_uint8_bgr(image)
        boxes = [bbox[:] for bbox in bboxes]
        applied: list[str] = []

        if not self.config.enabled:
            return {"image": frame, "bboxes": clip_bboxes(boxes, frame.shape[1], frame.shape[0]), "applied": applied}

        frame, boxes, geo_applied = apply_detection_geometric(frame, boxes, self.config.det.geometric, rng)
        applied.extend(geo_applied)
        frame, photo_applied = apply_detection_photometric(frame, self.config.det.photometric, rng)
        applied.extend(photo_applied)
        boxes = clip_bboxes(boxes, frame.shape[1], frame.shape[0])
        return {"image": frame, "bboxes": boxes, "applied": applied}

    def run_rec(self, image: np.ndarray, text_type: str, seed: int | None = None) -> dict[str, Any]:
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        frame = ensure_uint8_bgr(image)
        original = frame.copy()
        applied: list[str] = []

        if not self.config.enabled:
            return {"image": frame, "applied": applied}

        frame, geo_applied = apply_recognition_geometric(frame, self.config.rec.geometric, text_type, rng)
        applied.extend(geo_applied)
        frame, photo_applied = apply_recognition_photometric(frame, self.config.rec.photometric, text_type, rng)
        applied.extend(photo_applied)

        if text_type == "single_char" and pixel_variance(frame) < 15.0:
            return {"image": original, "applied": ["fallback_original_single_char"]}
        return {"image": frame, "applied": applied}

    def run_batch_det(self, image_list: list[np.ndarray], bboxes_list: list[list[list[int]]]) -> list[dict[str, Any]]:
        return [self.run_det(image, boxes, seed=self.config.seed + index) for index, (image, boxes) in enumerate(zip(image_list, bboxes_list))]

    def run_batch_rec(self, image_list: list[np.ndarray], text_type_list: list[str]) -> list[dict[str, Any]]:
        return [self.run_rec(image, text_type, seed=self.config.seed + index) for index, (image, text_type) in enumerate(zip(image_list, text_type_list))]

    def visualize_det(self, image: np.ndarray, bboxes: list[list[int]], n_samples: int) -> np.ndarray:
        panels: list[np.ndarray] = []
        for index in range(n_samples):
            result = self.run_det(image, bboxes, seed=self.config.seed + index)
            panels.append(draw_bboxes(result["image"], result["bboxes"], ",".join(result["applied"]) or "none"))
        return stack_panels(panels)

    def visualize_rec(self, image: np.ndarray, text_type: str, n_samples: int) -> np.ndarray:
        panels: list[np.ndarray] = []
        for index in range(n_samples):
            result = self.run_rec(image, text_type, seed=self.config.seed + index)
            panels.append(annotate_image(result["image"], ",".join(result["applied"]) or "none"))
        return stack_panels(panels)


def load_augment_config(config_path: str | Path) -> AugmentationConfig:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    aug = payload.get("augmentation", {})
    det = aug.get("det", {})
    rec = aug.get("rec", {})
    det_geo = det.get("geometric", {})
    det_photo = det.get("photometric", {})
    rec_geo = rec.get("geometric", {})
    rec_photo = rec.get("photometric", {})
    return AugmentationConfig(
        enabled=aug.get("enabled", True),
        seed=aug.get("seed", 42),
        det=DetectionAugConfig(
            geometric=DetectionGeometricConfig(
                rotate=RangeConfig(**det_geo.get("rotate", {"range": [-10, 10]})),
                random_crop=CropConfig(**det_geo.get("random_crop", {})),
                horizontal_flip=ToggleConfig(**det_geo.get("horizontal_flip", {})),
                perspective=PerspectiveConfig(**det_geo.get("perspective", {})),
                resize_jitter=RangeConfig(**det_geo.get("resize_jitter", {"range": [0.5, 2.0]})),
            ),
            photometric=DetectionPhotometricConfig(
                brightness=RangeConfig(**det_photo.get("brightness", {"range": [0.6, 1.4]})),
                contrast=RangeConfig(**det_photo.get("contrast", {"range": [0.7, 1.5]})),
                hsv_shift=ToggleConfig(**det_photo.get("hsv_shift", {})),
                gaussian_noise=SigmaConfig(**det_photo.get("gaussian_noise", {"sigma_range": [1, 8]})),
                motion_blur=KernelConfig(**det_photo.get("motion_blur", {"kernel_range": [3, 5]})),
                jpeg_quality=QualityConfig(**det_photo.get("jpeg_quality", {"quality_range": [50, 95]})),
                shadow=ToggleConfig(**det_photo.get("shadow", {})),
            ),
        ),
        rec=RecognitionAugConfig(
            geometric=RecognitionGeometricConfig(
                rotate=RangeConfig(**rec_geo.get("rotate", {"range": [-3, 3]})),
                stretch_x=RangeConfig(**rec_geo.get("stretch_x", {"range": [0.9, 1.1]})),
                stretch_y=RangeConfig(**rec_geo.get("stretch_y", {"range": [0.9, 1.05]})),
            ),
            photometric=RecognitionPhotometricConfig(
                brightness=RangeConfig(**rec_photo.get("brightness", {"range": [0.5, 1.5]})),
                contrast=RangeConfig(**rec_photo.get("contrast", {"range": [0.6, 1.6]})),
                gaussian_noise=SigmaConfig(**rec_photo.get("gaussian_noise", {"sigma_range": [1, 5]})),
                gaussian_blur=SigmaConfig(**rec_photo.get("gaussian_blur", {"sigma_range": [0.3, 1.2]})),
                jpeg_quality=QualityConfig(**rec_photo.get("jpeg_quality", {"quality_range": [40, 90]})),
                dilate=KernelConfig(**rec_photo.get("dilate", {"kernel_range": [1, 2]})),
                erode=KernelConfig(**rec_photo.get("erode", {"kernel_range": [1, 2]})),
                grayscale=ToggleConfig(**rec_photo.get("grayscale", {})),
            ),
        ),
    )


def describe_augmentation_pipeline(config: AugmentationConfig | None = None) -> list[str]:
    config = config or AugmentationConfig()
    return [
        f"enabled={config.enabled}",
        f"seed={config.seed}",
        "det: geometric_then_photometric",
        "rec: light_geometric_then_photometric",
    ]


def apply_detection_geometric(
    image: np.ndarray,
    bboxes: list[list[int]],
    config: DetectionGeometricConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[list[int]], list[str]]:
    frame = image
    boxes = [bbox[:] for bbox in bboxes]
    applied: list[str] = []

    if should_apply(config.rotate, rng):
        frame, boxes = rotate_with_bboxes(frame, boxes, sample_float(config.rotate.range, rng))
        applied.append("rotate")
    if should_apply(config.random_crop, rng):
        frame, boxes, changed = random_crop_with_bboxes(frame, boxes, config.random_crop.min_area_ratio, config.random_crop.bbox_overlap_thresh, rng)
        if changed:
            applied.append("random_crop")
    if should_apply(config.horizontal_flip, rng):
        frame, boxes = horizontal_flip_with_bboxes(frame, boxes)
        applied.append("horizontal_flip")
    if should_apply(config.perspective, rng):
        frame, boxes = perspective_with_bboxes(frame, boxes, config.perspective.distort_limit, rng)
        applied.append("perspective")
    if should_apply(config.resize_jitter, rng):
        frame, boxes = resize_jitter_with_bboxes(frame, boxes, sample_float(config.resize_jitter.range, rng))
        applied.append("resize_jitter")

    return frame, boxes, applied


def apply_detection_photometric(
    image: np.ndarray,
    config: DetectionPhotometricConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    frame = image
    applied: list[str] = []

    if should_apply(config.brightness, rng):
        frame = adjust_brightness(frame, sample_float(config.brightness.range, rng))
        applied.append("brightness")
    if should_apply(config.contrast, rng):
        frame = adjust_contrast(frame, sample_float(config.contrast.range, rng))
        applied.append("contrast")
    if should_apply(config.hsv_shift, rng):
        frame = apply_hsv_shift(frame, rng)
        applied.append("hsv_shift")
    if should_apply(config.gaussian_noise, rng):
        frame = add_gaussian_noise(frame, sample_float(config.gaussian_noise.sigma_range, rng), rng)
        applied.append("gaussian_noise")
    if should_apply(config.motion_blur, rng):
        frame = apply_motion_blur(frame, sample_int(config.motion_blur.kernel_range, rng), rng)
        applied.append("motion_blur")
    if should_apply(config.jpeg_quality, rng):
        frame = apply_jpeg_artifact(frame, sample_int(config.jpeg_quality.quality_range, rng))
        applied.append("jpeg_quality")
    if should_apply(config.shadow, rng):
        frame = apply_shadow(frame, rng)
        applied.append("shadow")

    return frame, applied


def apply_recognition_geometric(
    image: np.ndarray,
    config: RecognitionGeometricConfig,
    text_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    frame = image
    applied: list[str] = []
    strength = recognition_strength_multiplier(text_type)

    if should_apply(config.rotate, rng):
        frame = rotate_image(frame, sample_float(scale_range(config.rotate.range, strength), rng))
        applied.append("rotate")
    if text_type != "date" and should_apply(config.stretch_x, rng):
        frame = stretch_image(frame, sample_float(scale_range(config.stretch_x.range, strength), rng), 1.0)
        applied.append("stretch_x")
    if should_apply(config.stretch_y, rng):
        frame = stretch_image(frame, 1.0, sample_float(scale_range(config.stretch_y.range, strength), rng))
        applied.append("stretch_y")

    return frame, applied


def apply_recognition_photometric(
    image: np.ndarray,
    config: RecognitionPhotometricConfig,
    text_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    frame = image
    applied: list[str] = []
    strength = recognition_strength_multiplier(text_type)
    photo_prob = 0.5 if text_type == "handwrite" else 1.0

    if should_apply_prob(config.brightness, photo_prob, rng):
        frame = adjust_brightness(frame, sample_float(scale_range(config.brightness.range, strength), rng))
        applied.append("brightness")
    if should_apply_prob(config.contrast, photo_prob, rng):
        frame = adjust_contrast(frame, sample_float(scale_range(config.contrast.range, strength), rng))
        applied.append("contrast")
    if should_apply_prob(config.gaussian_noise, photo_prob, rng):
        frame = add_gaussian_noise(frame, sample_float(scale_range(config.gaussian_noise.sigma_range, strength), rng), rng)
        applied.append("gaussian_noise")
    if should_apply_prob(config.gaussian_blur, photo_prob, rng):
        frame = apply_gaussian_blur(frame, sample_float(scale_range(config.gaussian_blur.sigma_range, strength), rng))
        applied.append("gaussian_blur")
    jpeg_prob = 0.9 if text_type == "date" else 1.0
    if should_apply_prob(config.jpeg_quality, jpeg_prob, rng):
        frame = apply_jpeg_artifact(frame, sample_int(config.jpeg_quality.quality_range, rng))
        applied.append("jpeg_quality")

    dilate_on = should_apply_prob(config.dilate, photo_prob, rng)
    erode_on = False if dilate_on else should_apply_prob(config.erode, photo_prob, rng)
    if dilate_on:
        frame = apply_morphology(frame, "dilate", sample_int(config.dilate.kernel_range, rng))
        applied.append("dilate")
    elif erode_on:
        frame = apply_morphology(frame, "erode", sample_int(config.erode.kernel_range, rng))
        applied.append("erode")

    if should_apply(config.grayscale, rng):
        frame = to_grayscale_bgr(frame)
        applied.append("grayscale")

    return frame, applied


def ensure_uint8_bgr(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array.copy()


def should_apply(config: Any, rng: np.random.Generator) -> bool:
    return bool(getattr(config, "enabled", True) and rng.random() < getattr(config, "prob", 0.0))


def should_apply_prob(config: Any, multiplier: float, rng: np.random.Generator) -> bool:
    return bool(getattr(config, "enabled", True) and rng.random() < getattr(config, "prob", 0.0) * multiplier)


def sample_float(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def sample_int(bounds: tuple[int, int], rng: np.random.Generator) -> int:
    return int(rng.integers(bounds[0], bounds[1] + 1))


def scale_range(bounds: tuple[float, float], strength: float) -> tuple[float, float]:
    center = (bounds[0] + bounds[1]) / 2.0
    half = (bounds[1] - bounds[0]) / 2.0 * strength
    return (center - half, center + half)


def recognition_strength_multiplier(text_type: str) -> float:
    if text_type == "single_char":
        return 0.5
    if text_type == "handwrite":
        return 0.7
    return 1.0


def rotate_with_bboxes(image: np.ndarray, bboxes: list[list[int]], angle: float) -> tuple[np.ndarray, list[list[int]]]:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    transformed = [transform_bbox_affine(bbox, matrix, width, height) for bbox in bboxes]
    return rotated, transformed


def random_crop_with_bboxes(
    image: np.ndarray,
    bboxes: list[list[int]],
    min_area_ratio: float,
    bbox_overlap_thresh: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[list[int]], bool]:
    height, width = image.shape[:2]
    ratio = sample_float((min_area_ratio, 1.0), rng)
    crop_h = max(1, int(height * ratio))
    crop_w = max(1, int(width * ratio))
    if crop_h >= height and crop_w >= width:
        return image, bboxes, False

    top = int(rng.integers(0, max(1, height - crop_h + 1)))
    left = int(rng.integers(0, max(1, width - crop_w + 1)))
    crop_region = [left, top, left + crop_w, top + crop_h]
    kept: list[list[int]] = []
    for bbox in bboxes:
        clipped = intersect_bbox(bbox, crop_region)
        if clipped is None:
            continue
        if bbox_area(clipped) / max(1.0, bbox_area(bbox)) >= bbox_overlap_thresh:
            kept.append([clipped[0] - left, clipped[1] - top, clipped[2] - left, clipped[3] - top])
    return image[top : top + crop_h, left : left + crop_w], kept, True


def horizontal_flip_with_bboxes(image: np.ndarray, bboxes: list[list[int]]) -> tuple[np.ndarray, list[list[int]]]:
    width = image.shape[1]
    flipped = cv2.flip(image, 1)
    return flipped, [[width - bbox[2], bbox[1], width - bbox[0], bbox[3]] for bbox in bboxes]


def perspective_with_bboxes(image: np.ndarray, bboxes: list[list[int]], distort_limit: float, rng: np.random.Generator) -> tuple[np.ndarray, list[list[int]]]:
    height, width = image.shape[:2]
    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    dx = width * distort_limit
    dy = height * distort_limit
    dst = src + np.array(
        [
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    transformed = [transform_bbox_perspective(bbox, matrix, width, height) for bbox in bboxes]
    return warped, transformed


def resize_jitter_with_bboxes(image: np.ndarray, bboxes: list[list[int]], scale: float) -> tuple[np.ndarray, list[list[int]]]:
    height, width = image.shape[:2]
    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    sx = target_w / width
    sy = target_h / height
    boxes = [[int(round(bbox[0] * sx)), int(round(bbox[1] * sy)), int(round(bbox[2] * sx)), int(round(bbox[3] * sy))] for bbox in bboxes]
    return resized, boxes


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def apply_hsv_shift(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.8, 1.2), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, kernel_size: int, rng: np.random.Generator) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((kernel_size / 2.0, kernel_size / 2.0), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel /= max(np.sum(kernel), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_jpeg_artifact(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def apply_shadow(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    alpha = rng.uniform(0.15, 0.35)
    if rng.random() < 0.5:
        ramp = np.linspace(1.0 - alpha, 1.0, width, dtype=np.float32)
        mask = np.tile(ramp, (height, 1))
    else:
        ramp = np.linspace(1.0 - alpha, 1.0, height, dtype=np.float32)
        mask = np.tile(ramp[:, None], (1, width))
    return np.clip(image.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)


def apply_morphology(image: np.ndarray, op: str, kernel_size: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if op == "dilate":
        out = cv2.dilate(gray, kernel, iterations=1)
    else:
        out = cv2.erode(gray, kernel, iterations=1)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def to_grayscale_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def stretch_image(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (max(1, int(round(width * scale_x))), max(1, int(round(height * scale_y)))), interpolation=cv2.INTER_CUBIC)


def transform_bbox_affine(bbox: list[int], matrix: np.ndarray, width: int, height: int) -> list[int]:
    points = np.array([[bbox[0], bbox[1], 1], [bbox[2], bbox[1], 1], [bbox[2], bbox[3], 1], [bbox[0], bbox[3], 1]], dtype=np.float32)
    transformed = (matrix @ points.T).T
    xs = np.clip(transformed[:, 0], 0, width - 1)
    ys = np.clip(transformed[:, 1], 0, height - 1)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def transform_bbox_perspective(bbox: list[int], matrix: np.ndarray, width: int, height: int) -> list[int]:
    points = np.array([[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(points, matrix)[0]
    xs = np.clip(transformed[:, 0], 0, width - 1)
    ys = np.clip(transformed[:, 1], 0, height - 1)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def clip_bboxes(bboxes: list[list[int]], width: int, height: int) -> list[list[int]]:
    clipped: list[list[int]] = []
    for bbox in bboxes:
        x1 = int(np.clip(bbox[0], 0, width - 1))
        y1 = int(np.clip(bbox[1], 0, height - 1))
        x2 = int(np.clip(bbox[2], 0, width - 1))
        y2 = int(np.clip(bbox[3], 0, height - 1))
        if x2 > x1 and y2 > y1:
            clipped.append([x1, y1, x2, y2])
    return clipped


def intersect_bbox(bbox: list[int], crop: list[int]) -> list[int] | None:
    x1 = max(bbox[0], crop[0])
    y1 = max(bbox[1], crop[1])
    x2 = min(bbox[2], crop[2])
    y2 = min(bbox[3], crop[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_area(bbox: list[int]) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def pixel_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.var(gray.astype(np.float32)))


def annotate_image(image: np.ndarray, title: str) -> np.ndarray:
    panel = image.copy()
    cv2.putText(panel, title[:48], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return panel


def draw_bboxes(image: np.ndarray, bboxes: list[list[int]], title: str) -> np.ndarray:
    panel = annotate_image(image, title)
    for bbox in bboxes:
        cv2.rectangle(panel, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 180, 0), 2)
    return panel


def stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    if not panels:
        raise ValueError("no panels to stack")
    max_height = max(panel.shape[0] for panel in panels)
    resized: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] != max_height:
            scale = max_height / panel.shape[0]
            width = max(1, int(round(panel.shape[1] * scale)))
            panel = cv2.resize(panel, (width, max_height), interpolation=cv2.INTER_CUBIC)
        resized.append(panel)
    return np.concatenate(resized, axis=1)
