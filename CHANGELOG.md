# Changelog

## [v1.0.0] - 2026-04-22

### Changed

- Added `ocr_service_v2.py` as the ShiftFlow OCR service integration target before replacing a legacy `ocr_service.py`.
- Replaced the planned PaddleOCR default model call path with a staged Cls -> Det -> Rec predictor pipeline.
- Added legacy-compatible OCR response normalization for `/ocr` style responses.

### Added

- Singleton OCR service loading helpers.
- `ENABLE_OCR=false` compatible empty OCR responses.
- Bounding-box restore and clipping helpers for padded/resized Detection inputs.
- Optional FastAPI app factory for `/ocr` and `/roster/parse` contract testing.
- OCR-S02 unit tests for service contract, singleton behavior, response normalization, and box restoration.
- OCR-S03 confidence levels, cell IDs, confidence summaries, feedback JSONL logging, feedback stats, and feedback dataset export.
- OCR-S04 model registry, active version tracking, deployment/rollback scripts, deployment history, version API helpers, and rollback trigger rules.
