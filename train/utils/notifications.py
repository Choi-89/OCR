from __future__ import annotations

import os
from typing import Any


def send_slack_notification(message: str, webhook_env: str = "SLACK_WEBHOOK_URL") -> dict[str, Any]:
    webhook_url = os.environ.get(webhook_env)
    if not webhook_url:
        return {"sent": False, "reason": "missing_webhook"}
    try:
        import requests

        response = requests.post(webhook_url, json={"text": message}, timeout=10)
        return {"sent": response.ok, "status_code": response.status_code}
    except Exception as exc:
        return {"sent": False, "reason": str(exc)}


def build_completion_message(experiment_name: str, best_metric: str, best_epoch: int, checkpoint_path: str) -> str:
    return "\n".join(
        [
            f"Training completed: {experiment_name}",
            f"Best metric: {best_metric} (epoch {best_epoch})",
            f"Checkpoint: {checkpoint_path}",
        ]
    )
