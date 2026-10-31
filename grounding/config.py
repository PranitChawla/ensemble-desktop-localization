from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path = "models.json") -> dict:
    config = json.loads(Path(path).read_text())
    models = config["models"]
    if len(models) != 3:
        raise ValueError("This task requires exactly three frozen models")
    if len({m["alias"] for m in models}) != len(models):
        raise ValueError("Model aliases must be unique")
    if len({m["port"] for m in models}) != len(models):
        raise ValueError("Server ports must be unique")
    if sum(m["gpu_memory_utilization"] for m in models) > 0.8:
        raise ValueError("Reserve at least 20% GPU headroom")
    for model in models:
        if not 0 < model["gpu_memory_utilization"] < 1:
            raise ValueError("GPU memory fractions must be between zero and one")
        if not 1 <= model["port"] <= 65535:
            raise ValueError("Invalid model server port")
        if len(model["revision"]) != 40 or any(c not in "0123456789abcdef" for c in model["revision"]):
            raise ValueError("Pin every model to an immutable Hugging Face commit")
        convention = model.get("grounding", {})
        if convention.get("request_space") not in {"pixels", "norm1000", "norm999", "norm01"}:
            raise ValueError("Declare each model's requested grounding coordinate space")
    return config
