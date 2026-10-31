from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path = "models.json") -> dict:
    config = json.loads(Path(path).read_text())
    models = config["models"]
    if len({m["alias"] for m in models}) != len(models):
        raise ValueError("Model aliases must be unique")
    if len({m["port"] for m in models}) != len(models):
        raise ValueError("Server ports must be unique")
    if sum(m["gpu_memory_utilization"] for m in models) > 0.8:
        raise ValueError("Reserve at least 20% GPU headroom")
    for model in models:
        if len(model["revision"]) != 40:
            raise ValueError("Pin every model to an immutable Hugging Face commit")
    return config

