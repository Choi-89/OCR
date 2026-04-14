from __future__ import annotations

import logging
from pathlib import Path


def build_logger(name: str, log_dir: str | Path) -> logging.Logger:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(directory / f"{name}.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
