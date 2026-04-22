from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_reproducible_seed(seed: int = 42) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    paddle_status = "not_available"
    try:
        import paddle

        paddle.seed(seed)
        paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        paddle_status = "seeded"
    except Exception:
        paddle_status = "not_available"
    return {"seed": seed, "paddle": paddle_status, "cudnn_deterministic": paddle_status == "seeded"}
