from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable


ModelBuilder = Callable[[dict[str, Any]], Any]


@dataclass(slots=True)
class InferenceExportSpec:
    model_key: str
    model_name: str
    config_path: Path
    checkpoint_path: Path
    output_dir: Path
    input_shape: list[int | None]
    input_dtype: str = "float32"
    dict_path: Path | None = None
    performance: dict[str, float] = field(default_factory=dict)
    quantized: bool = False


@dataclass(slots=True)
class ExportResult:
    model_key: str
    output_dir: Path
    exported_files: list[Path]
    model_info_path: Path
    dry_run: bool


def default_export_specs(root: str | Path = ".") -> dict[str, InferenceExportSpec]:
    base = Path(root)
    return {
        "det": InferenceExportSpec(
            model_key="det",
            model_name="DBNet++",
            config_path=base / "train/configs/det_config.yaml",
            checkpoint_path=base / "train/checkpoints/det/best.pdparams",
            output_dir=base / "backend/ocr/inference/det",
            input_shape=[None, 3, None, None],
        ),
        "rec": InferenceExportSpec(
            model_key="rec",
            model_name="SVTR-Tiny",
            config_path=base / "train/configs/rec_config.yaml",
            checkpoint_path=base / "train/checkpoints/rec/best.pdparams",
            output_dir=base / "backend/ocr/inference/rec",
            input_shape=[None, 3, 32, None],
            dict_path=base / "backend/ocr/dict/dict_latest.txt",
        ),
        "cls": InferenceExportSpec(
            model_key="cls",
            model_name="MobileNetV3Small",
            config_path=base / "train/configs/cls_config.yaml",
            checkpoint_path=base / "train/checkpoints/cls/best.pdparams",
            output_dir=base / "backend/ocr/inference/cls",
            input_shape=[None, 3, 48, 192],
        ),
    }


def initialize_inference_workspace(root: str | Path) -> dict[str, str]:
    base = Path(root) / "backend/ocr/inference"
    paths = {
        "det": base / "det",
        "rec": base / "rec",
        "cls": base / "cls",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {key: str(path) for key, path in paths.items()}


def export_inference_model(
    spec: InferenceExportSpec,
    model_builder: ModelBuilder | None = None,
    dry_run: bool = False,
) -> ExportResult:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    copied_files = copy_sidecar_files(spec)
    if not dry_run:
        if model_builder is None:
            raise RuntimeError("A real Paddle model_builder is required when dry_run=False.")
        export_with_paddle(spec, model_builder)
    model_info_path = write_model_info(spec)
    exported_files = expected_inference_files(spec.output_dir)
    if dry_run:
        exported_files = copied_files + [model_info_path]
    return ExportResult(
        model_key=spec.model_key,
        output_dir=spec.output_dir,
        exported_files=exported_files,
        model_info_path=model_info_path,
        dry_run=dry_run,
    )


def export_all_models(
    root: str | Path = ".",
    dry_run: bool = False,
    builders: dict[str, ModelBuilder] | None = None,
) -> dict[str, ExportResult]:
    specs = default_export_specs(root)
    return {
        key: export_inference_model(spec, model_builder=(builders or {}).get(key), dry_run=dry_run)
        for key, spec in specs.items()
    }


def export_with_paddle(spec: InferenceExportSpec, model_builder: ModelBuilder) -> None:
    try:
        import paddle
    except ImportError as exc:
        raise RuntimeError("PaddlePaddle is required for real inference export.") from exc

    config = read_config(spec.config_path)
    model = model_builder(config)
    state = paddle.load(str(spec.checkpoint_path))
    weights = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.set_state_dict(weights)
    model.eval()
    input_spec = [paddle.static.InputSpec(shape=spec.input_shape, dtype=spec.input_dtype, name="image")]
    static_model = paddle.jit.to_static(model, input_spec=input_spec)
    paddle.jit.save(static_model, str(spec.output_dir / "inference"))


def copy_sidecar_files(spec: InferenceExportSpec) -> list[Path]:
    copied: list[Path] = []
    if spec.dict_path and spec.dict_path.exists():
        target = spec.output_dir / "dict.txt"
        shutil.copyfile(spec.dict_path, target)
        copied.append(target)
    return copied


def write_model_info(spec: InferenceExportSpec) -> Path:
    info = {
        "model_name": spec.model_name,
        "model_key": spec.model_key,
        "version": "v1.0.0",
        "export_date": str(date.today()),
        "source_checkpoint": str(spec.checkpoint_path),
        "config_path": str(spec.config_path),
        "input_spec": {"shape": spec.input_shape, "dtype": spec.input_dtype},
        "performance": spec.performance,
        "quantized": spec.quantized,
    }
    path = spec.output_dir / "model_info.json"
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def expected_inference_files(output_dir: str | Path) -> list[Path]:
    base = Path(output_dir)
    return [base / "inference.pdmodel", base / "inference.pdiparams", base / "inference.pdiparams.info", base / "model_info.json"]


def verify_export_outputs(output_dir: str | Path, require_binary: bool = True) -> dict[str, Any]:
    files = expected_inference_files(output_dir)
    required = files if require_binary else [Path(output_dir) / "model_info.json"]
    missing = [str(path) for path in required if not path.exists()]
    return {"output_dir": str(output_dir), "passed": not missing, "missing": missing}


def model_equivalence_report(max_abs_diff: float, tolerance: float = 1e-5) -> dict[str, Any]:
    return {"max_abs_diff": max_abs_diff, "tolerance": tolerance, "passed": max_abs_diff <= tolerance}


def metric_delta_report(before: float, after: float, tolerance: float, lower_is_better: bool = False) -> dict[str, Any]:
    delta = after - before
    abs_delta = abs(delta)
    return {
        "before": before,
        "after": after,
        "delta": delta,
        "tolerance": tolerance,
        "lower_is_better": lower_is_better,
        "passed": abs_delta <= tolerance,
    }


def measure_latency(callable_fn: Callable[[], Any], warmup: int = 10, repeat: int = 100) -> dict[str, float]:
    for _ in range(warmup):
        callable_fn()
    timings: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        callable_fn()
        timings.append(time.perf_counter() - start)
    timings.sort()
    return {
        "p50": timings[int((repeat - 1) * 0.50)],
        "p95": timings[int((repeat - 1) * 0.95)],
        "p99": timings[int((repeat - 1) * 0.99)],
        "mean": sum(timings) / len(timings),
        "max": max(timings),
    }


def build_predictor_config(model_dir: str | Path, use_gpu: bool = False, device_id: int = 0, cpu_threads: int = 4) -> dict[str, Any]:
    model_path = Path(model_dir) / "inference.pdmodel"
    params_path = Path(model_dir) / "inference.pdiparams"
    return {
        "model_file": str(model_path),
        "params_file": str(params_path),
        "use_gpu": use_gpu,
        "device_id": device_id,
        "cpu_threads": cpu_threads,
        "memory_optim": True,
        "ir_optim": True,
    }


def build_predictor(model_dir: str | Path, use_gpu: bool = False, device_id: int = 0, cpu_threads: int = 4) -> Any:
    try:
        from paddle.inference import Config, create_predictor
    except ImportError as exc:
        raise RuntimeError("PaddlePaddle inference runtime is required to build a predictor.") from exc

    cfg = build_predictor_config(model_dir, use_gpu=use_gpu, device_id=device_id, cpu_threads=cpu_threads)
    config = Config(cfg["model_file"], cfg["params_file"])
    if use_gpu:
        config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=device_id)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    return create_predictor(config)


def read_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    return yaml.safe_load(source.read_text(encoding="utf-8")) or {}


def export_result_as_dict(result: ExportResult) -> dict[str, Any]:
    return {
        "model_key": result.model_key,
        "output_dir": str(result.output_dir),
        "exported_files": [str(path) for path in result.exported_files],
        "model_info_path": str(result.model_info_path),
        "dry_run": result.dry_run,
    }


def spec_as_dict(spec: InferenceExportSpec) -> dict[str, Any]:
    payload = asdict(spec)
    for key in ("config_path", "checkpoint_path", "output_dir", "dict_path"):
        if payload.get(key) is not None:
            payload[key] = str(payload[key])
    return payload
