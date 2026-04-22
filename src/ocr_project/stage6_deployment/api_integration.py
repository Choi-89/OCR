from __future__ import annotations

from typing import Any

import numpy as np

from ocr_project.stage6_deployment.confidence_ui import (
    FeedbackStoreConfig,
    OCRFeedbackRequest,
    confidence_config_from_env,
    enrich_ocr_response,
    feedback_stats,
    submit_feedback,
)
from ocr_project.stage6_deployment.ocr_service_v2 import OCRService, get_ocr_service
from ocr_project.stage6_deployment.versioning import deployment_history, health_payload, version_api_payload


def normalize_ocr_api_response(prediction: dict[str, Any], *, import_id: str = "imp_pending") -> dict[str, Any]:
    base = {
        "results": [
            {
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 0.0),
                "bbox": item.get("bbox", []),
                **({"cell_id": item["cell_id"]} if "cell_id" in item else {}),
            }
            for item in prediction.get("results", [])
        ],
        "processing_time": prediction.get("processing_time", 0.0),
        "enabled": prediction.get("enabled", True),
        "angle_corrected": prediction.get("angle_corrected", False),
    }
    return enrich_ocr_response(base, import_id=import_id, config=confidence_config_from_env())


def parse_roster_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    response = normalize_ocr_api_response(prediction)
    return {
        "entries": response["results"],
        "ocr": response,
    }


def predict_array_for_api(image: np.ndarray, service: OCRService | None = None, *, import_id: str = "imp_pending") -> dict[str, Any]:
    active_service = service or get_ocr_service()
    return normalize_ocr_api_response(active_service.predict(image), import_id=import_id)


def create_fastapi_app(service: OCRService | None = None) -> Any:
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
    except ImportError as exc:
        raise RuntimeError("FastAPI is required to create the OCR API app.") from exc

    app = FastAPI()

    @app.on_event("startup")
    def startup() -> None:
        get_ocr_service()

    @app.post("/ocr")
    async def ocr_endpoint(file: UploadFile = File(...), import_id: str = "imp_pending") -> dict[str, Any]:
        image = await decode_upload_to_image(file)
        return predict_array_for_api(image, service, import_id=import_id)

    @app.post("/roster/parse")
    async def roster_parse_endpoint(file: UploadFile = File(...), import_id: str = "imp_pending") -> dict[str, Any]:
        image = await decode_upload_to_image(file)
        return parse_roster_payload(predict_array_for_api(image, service, import_id=import_id))

    @app.post("/ocr/feedback")
    async def ocr_feedback_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        return submit_ocr_feedback(payload)

    @app.get("/admin/ocr/feedback-stats")
    async def feedback_stats_endpoint() -> dict[str, Any]:
        return get_feedback_stats()

    @app.get("/ocr/version")
    async def ocr_version_endpoint() -> dict[str, Any]:
        return get_ocr_version()

    @app.get("/admin/ocr/deployment-history")
    async def deployment_history_endpoint(limit: int = 20) -> dict[str, Any]:
        return get_deployment_history(limit)

    @app.get("/health")
    async def health_endpoint() -> dict[str, Any]:
        return get_health_payload()

    async def decode_upload_to_image(file: UploadFile) -> np.ndarray:
        data = await file.read()
        image = decode_image_bytes(data)
        if image is None:
            raise HTTPException(status_code=400, detail="invalid image file")
        return image

    return app


def submit_ocr_feedback(payload: dict[str, Any], *, store_config: FeedbackStoreConfig | None = None, worker_id: str | None = None) -> dict[str, Any]:
    return submit_feedback(OCRFeedbackRequest(**payload), store_config=store_config, worker_id=worker_id)


def get_feedback_stats(store_config: FeedbackStoreConfig | None = None, retrain_threshold: int = 100) -> dict[str, Any]:
    cfg = store_config or FeedbackStoreConfig()
    return feedback_stats(cfg.root / cfg.log_name, retrain_threshold=retrain_threshold)


def get_ocr_version() -> dict[str, Any]:
    return version_api_payload()


def get_deployment_history(limit: int = 20) -> dict[str, Any]:
    return deployment_history(limit=limit)


def get_health_payload(error_rate_1h: float = 0.0, avg_response_ms: float = 0.0, ocr_enabled: bool = True) -> dict[str, Any]:
    return health_payload(error_rate_1h=error_rate_1h, avg_response_ms=avg_response_ms, ocr_enabled=ocr_enabled)


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image
