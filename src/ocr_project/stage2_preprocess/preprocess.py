from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import yaml


IMREAD_UNCHANGED = cv2.IMREAD_UNCHANGED


@dataclass(slots=True)
class LoadConfig:
    enabled: bool = True
    white_background: bool = True


@dataclass(slots=True)
class AngleCorrectionConfig:
    enabled: bool = True
    method: str = "classifier"
    fallback: str = "exif"


@dataclass(slots=True)
class DenoiseConfig:
    enabled: bool = True
    laplacian_threshold: float = 100.0
    h: int = 10
    hColor: int = 10
    templateWindowSize: int = 7
    searchWindowSize: int = 21


@dataclass(slots=True)
class BinarizeConfig:
    enabled: bool = False
    method: str = "adaptive"


@dataclass(slots=True)
class SharpenConfig:
    enabled: bool = True
    laplacian_threshold: float = 500.0


@dataclass(slots=True)
class DeskewConfig:
    enabled: bool = True
    method: str = "hough"
    min_angle: float = 0.5
    max_angle: float = 30.0
    interpolation: str = "cubic"
    projection_step: float = 0.5


@dataclass(slots=True)
class ResizeConfig:
    det_max_side: int = 960
    rec_height: int = 32
    rec_max_width: int = 320
    padding_color: int = 255


@dataclass(slots=True)
class NormalizeConfig:
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class PreprocessConfig:
    load: LoadConfig = field(default_factory=LoadConfig)
    angle_correction: AngleCorrectionConfig = field(default_factory=AngleCorrectionConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    binarize: BinarizeConfig = field(default_factory=BinarizeConfig)
    sharpen: SharpenConfig = field(default_factory=SharpenConfig)
    deskew: DeskewConfig = field(default_factory=DeskewConfig)
    resize: ResizeConfig = field(default_factory=ResizeConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass(slots=True)
class PreprocessMetadata:
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    padding: tuple[int, int, int, int]
    angle_corrected: bool
    deskew_angle: float
    exif_applied: bool
    load_warning: str | None = None
    processing_ms: float = 0.0


class PreprocessPipeline:
    """Shared preprocessing pipeline for both training and inference."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = load_preprocess_config(self.config_path)

    def run(
        self,
        image: str | Path | np.ndarray,
        mode: str,
        *,
        angle: int | None = None,
    ) -> dict[str, Any]:
        start = perf_counter()
        if mode not in {"det", "rec"}:
            raise ValueError(f"unsupported mode: {mode}")

        stages: dict[str, np.ndarray] = {}
        loaded = self._load_and_normalize(image)
        frame = loaded["image"]
        stages["loaded"] = frame.copy()
        original_shape = frame.shape[:2]

        frame, angle_corrected = self._apply_angle_correction(
            frame,
            manual_angle=angle,
            exif_applied=loaded["exif_applied"],
        )
        stages["angle_corrected"] = frame.copy()

        frame = self._apply_noise_and_sharpen(frame)
        stages["denoise_sharpen"] = frame.copy()

        frame, deskew_angle = self._apply_deskew(frame)
        stages["deskew"] = frame.copy()

        resized, padded_shape, padding = self._resize_for_mode(frame, mode)
        stages["resized"] = resized.copy()
        normalized = normalize_image(resized, self.config.normalize)

        elapsed_ms = (perf_counter() - start) * 1000.0
        metadata = PreprocessMetadata(
            original_shape=original_shape,
            padded_shape=padded_shape,
            padding=padding,
            angle_corrected=angle_corrected,
            deskew_angle=deskew_angle,
            exif_applied=loaded["exif_applied"],
            load_warning=loaded["warning"],
            processing_ms=elapsed_ms,
        )
        return {
            "image": normalized,
            "original_shape": metadata.original_shape,
            "padded_shape": metadata.padded_shape,
            "padding": metadata.padding,
            "angle_corrected": metadata.angle_corrected,
            "deskew_angle": metadata.deskew_angle,
            "exif_applied": metadata.exif_applied,
            "load_warning": metadata.load_warning,
            "processing_ms": metadata.processing_ms,
            "stages": stages,
        }

    def run_batch(
        self,
        image_list: list[str | Path | np.ndarray],
        mode: str,
    ) -> list[dict[str, Any]]:
        return [self.run(image, mode) for image in image_list]

    def visualize(
        self,
        image: str | Path | np.ndarray,
        mode: str,
        output_path: str | Path | None = None,
    ) -> np.ndarray:
        result = self.run(image, mode)
        panels = []
        for key in ("loaded", "angle_corrected", "denoise_sharpen", "deskew", "resized"):
            stage = result["stages"][key]
            panel = stage if stage.ndim == 3 else cv2.cvtColor(stage, cv2.COLOR_GRAY2BGR)
            panels.append(annotate_panel(panel, key))
        canvas = stack_panels(panels)
        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), canvas)
        return canvas

    def _load_and_normalize(self, image: str | Path | np.ndarray) -> dict[str, Any]:
        warning: str | None = None
        exif_applied = False

        if isinstance(image, np.ndarray):
            array = ensure_bgr_uint8(image)
        else:
            path = Path(image)
            if path.suffix.lower() == ".pdf":
                pages = load_pdf_pages(path)
                if not pages:
                    raise ValueError(f"failed to decode pdf: {path}")
                array = pages[0]
            else:
                raw = cv2.imread(str(path), IMREAD_UNCHANGED)
                if raw is None:
                    raise ValueError(f"failed to open image: {path}")
                array = raw

        array, exif_applied = apply_exif_orientation(array)
        array = ensure_bgr_uint8(array, white_background=self.config.load.white_background)

        if min(array.shape[:2]) < 720:
            warning = "resolution_below_720px"
        return {"image": array, "warning": warning, "exif_applied": exif_applied}

    def _apply_angle_correction(
        self,
        image: np.ndarray,
        *,
        manual_angle: int | None,
        exif_applied: bool,
    ) -> tuple[np.ndarray, bool]:
        cfg = self.config.angle_correction
        if not cfg.enabled:
            return image, False
        if manual_angle is not None:
            return rotate_quadrant(image, manual_angle), manual_angle % 360 != 0
        if exif_applied:
            return image, False

        method = cfg.method
        if method == "manual":
            return image, False
        if method == "exif":
            return image, False

        predicted_angle = estimate_orientation_180(image)
        if predicted_angle == 180:
            return rotate_quadrant(image, 180), True
        return image, False

    def _apply_noise_and_sharpen(self, image: np.ndarray) -> np.ndarray:
        frame = image.copy()
        laplacian_var = measure_laplacian_variance(frame)

        if self.config.denoise.enabled and laplacian_var < self.config.denoise.laplacian_threshold:
            cfg = self.config.denoise
            frame = cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                cfg.h,
                cfg.hColor,
                cfg.templateWindowSize,
                cfg.searchWindowSize,
            )

        if self.config.binarize.enabled:
            frame = apply_binarization(frame, self.config.binarize.method)

        laplacian_after = measure_laplacian_variance(frame)
        if self.config.sharpen.enabled and laplacian_after < self.config.sharpen.laplacian_threshold:
            frame = sharpen_image(frame)

        return frame

    def _apply_deskew(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        cfg = self.config.deskew
        if not cfg.enabled:
            return image, 0.0

        if cfg.method == "projection":
            angle = estimate_skew_projection(image, cfg.min_angle, cfg.max_angle, cfg.projection_step)
        else:
            angle = estimate_skew_hough(image)
            if angle is None:
                angle = estimate_skew_projection(image, cfg.min_angle, cfg.max_angle, cfg.projection_step)

        if angle is None:
            return image, 0.0
        if abs(angle) < cfg.min_angle:
            return image, 0.0
        if abs(angle) > cfg.max_angle:
            return image, float(angle)

        corrected = rotate_image(
            image,
            -float(angle),
            interpolation=cfg.interpolation,
            border_value=(255, 255, 255),
        )
        return corrected, float(angle)

    def _resize_for_mode(
        self,
        image: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
        if mode == "det":
            resized, padding = resize_for_detection(image, self.config.resize)
        else:
            resized, padding = resize_for_recognition(image, self.config.resize)
        return resized, resized.shape[:2], padding


def load_preprocess_config(config_path: str | Path) -> PreprocessConfig:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    preprocess = payload.get("preprocess", {})
    return PreprocessConfig(
        load=LoadConfig(**preprocess.get("load", {})),
        angle_correction=AngleCorrectionConfig(**preprocess.get("angle_correction", {})),
        denoise=DenoiseConfig(**preprocess.get("denoise", {})),
        binarize=BinarizeConfig(**preprocess.get("binarize", {})),
        sharpen=SharpenConfig(**preprocess.get("sharpen", {})),
        deskew=DeskewConfig(**preprocess.get("deskew", {})),
        resize=ResizeConfig(**preprocess.get("resize", {})),
        normalize=NormalizeConfig(**preprocess.get("normalize", {})),
    )


def ensure_bgr_uint8(image: np.ndarray, white_background: bool = True) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

    if array.ndim == 3 and array.shape[2] == 4:
        if white_background:
            alpha = array[:, :, 3:4].astype(np.float32) / 255.0
            rgb = array[:, :, :3].astype(np.float32)
            white = np.full_like(rgb, 255.0)
            composited = rgb * alpha + white * (1.0 - alpha)
            array = composited.astype(np.uint8)
        else:
            array = array[:, :, :3]
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    if array.ndim == 3 and array.shape[2] == 3:
        return array

    raise ValueError(f"unsupported image shape: {array.shape}")


def apply_exif_orientation(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """OpenCV strips EXIF in most cases; keep hook for PIL-based extension."""
    return image, False


def load_pdf_pages(path: Path) -> list[np.ndarray]:
    """Decode PDF through Pillow when available; return BGR images."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("pdf input requires Pillow support") from exc

    frames: list[np.ndarray] = []
    with Image.open(path) as pdf:
        page_count = getattr(pdf, "n_frames", 1)
        for page_index in range(page_count):
            pdf.seek(page_index)
            rgb = pdf.convert("RGB")
            array = np.array(rgb)
            frames.append(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    return frames


def rotate_quadrant(image: np.ndarray, angle: int) -> np.ndarray:
    normalized = angle % 360
    if normalized == 0:
        return image
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"angle must be 0/90/180/270, got {angle}")


def estimate_orientation_180(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    top_darkness = float(np.mean(gray[: max(1, gray.shape[0] // 4), :]))
    bottom_darkness = float(np.mean(gray[-max(1, gray.shape[0] // 4) :, :]))
    return 180 if bottom_darkness < top_darkness - 5 else 0


def measure_laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_binarization(image: np.ndarray, method: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def estimate_skew_hough(image: np.ndarray) -> float | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=100,
        minLineLength=max(40, image.shape[1] // 8),
        maxLineGap=20,
    )
    if lines is None:
        return None

    angles: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if -45.0 <= angle <= 45.0:
            angles.append(float(angle))
    if not angles:
        return None
    return float(np.median(np.array(angles, dtype=np.float32)))


def estimate_skew_projection(
    image: np.ndarray,
    min_angle: float = -15.0,
    max_angle: float = 15.0,
    step: float = 0.5,
) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best_angle = 0.0
    best_score = -1.0

    for angle in np.arange(min_angle, max_angle + step, step):
        rotated = rotate_image(binary, float(angle), interpolation="cubic", border_value=0)
        profile = np.sum(rotated, axis=1).astype(np.float32)
        score = float(np.var(profile))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def rotate_image(
    image: np.ndarray,
    angle: float,
    *,
    interpolation: str = "cubic",
    border_value: int | tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    interpolation_flag = interpolation_to_flag(interpolation)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def interpolation_to_flag(name: str) -> int:
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }
    return mapping.get(name, cv2.INTER_CUBIC)


def resize_for_detection(
    image: np.ndarray,
    config: ResizeConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    scale = min(config.det_max_side / max(height, width), 1.0)
    resized_w = max(32, int(round(width * scale)))
    resized_h = max(32, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    padded_w = int(np.ceil(resized_w / 32.0) * 32)
    padded_h = int(np.ceil(resized_h / 32.0) * 32)
    pad_right = padded_w - resized_w
    pad_bottom = padded_h - resized_h
    padded = cv2.copyMakeBorder(
        resized,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(config.padding_color, config.padding_color, config.padding_color),
    )
    return padded, (0, pad_bottom, 0, pad_right)


def resize_for_recognition(
    image: np.ndarray,
    config: ResizeConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    scale = config.rec_height / max(height, 1)
    resized_w = max(1, int(round(width * scale)))
    resized = cv2.resize(image, (resized_w, config.rec_height), interpolation=cv2.INTER_CUBIC)

    if resized_w > config.rec_max_width:
        resized = resized[:, : config.rec_max_width]
        return resized, (0, 0, 0, 0)

    pad_right = config.rec_max_width - resized.shape[1]
    padded = cv2.copyMakeBorder(
        resized,
        0,
        0,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(config.padding_color, config.padding_color, config.padding_color),
    )
    return padded, (0, 0, 0, pad_right)


def normalize_image(image: np.ndarray, config: NormalizeConfig) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(config.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(config.std, dtype=np.float32).reshape(1, 1, 3)
    return (rgb - mean) / std


def annotate_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = image.copy()
    cv2.putText(
        panel,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    if not panels:
        raise ValueError("no panels to stack")
    heights = [panel.shape[0] for panel in panels]
    target_height = max(heights)
    resized_panels = []
    for panel in panels:
        if panel.shape[0] != target_height:
            scale = target_height / panel.shape[0]
            width = max(1, int(round(panel.shape[1] * scale)))
            panel = cv2.resize(panel, (width, target_height), interpolation=cv2.INTER_CUBIC)
        resized_panels.append(panel)
    return np.concatenate(resized_panels, axis=1)
