from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_checkpoint(output_dir: str | Path, payload: dict[str, Any], filename: str = "latest.pdparams") -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
