"""Immutable task contract loaded by the evaluator, never by model output."""

from __future__ import annotations

import tomllib
from pathlib import Path

DEFAULT_TASK = Path(__file__).resolve().parent.parent / "task.toml"


def load_task(path=DEFAULT_TASK):
    task = tomllib.loads(Path(path).read_text())
    budget = task["budget"]
    for key in ["max_model_calls", "max_output_tokens_per_call", "max_images_per_request", "max_image_bytes"]:
        if type(budget[key]) is not int or budget[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["request_timeout_seconds", "case_timeout_seconds"]:
        if budget[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if budget["request_timeout_seconds"] > budget["case_timeout_seconds"]:
        raise ValueError("Request timeout must not exceed the case timeout")
    return task
