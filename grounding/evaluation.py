"""Visible development evaluation only. OSWorld-G is deliberately not loaded."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from PIL import Image

from grounding.client import ModelPool


def score_point(point, box, width, height) -> bool:
    if box is None:
        return point is None
    if point is None:
        return False
    x, y = point[0] * width, point[1] * height
    return bool(box[0] <= x <= box[2] and box[1] <= y <= box[3])


def load_candidate(path: str | Path):
    spec = importlib.util.spec_from_file_location("candidate_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(manifest: str | Path, candidate: str | Path, config: dict, output_dir: str | Path,
             *, max_calls=12, controller="qwen", limit: int | None = None) -> dict:
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty evaluation manifest")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    module = load_candidate(candidate)
    results = []
    for index, row in enumerate(rows):
        image_path = (manifest.parent / row["image"]).resolve()
        with Image.open(image_path) as image:
            width, height = image.size
        pool = ModelPool(config, max_calls=max_calls, trace_path=output_dir / f"case-{index:05d}.jsonl")
        started = time.monotonic()
        try:
            if hasattr(module,"ground"):
                prediction = module.ground(str(image_path), row["instruction"], pool, controller=controller)
            else:
                from grounding.program import run_program
                prediction=run_program(module.predict,str(image_path),row["instruction"],pool)
            completed = prediction["status"] == "completed"
            correct = completed and score_point(prediction["point"], row["bbox_xyxy"], width, height)
            result = {"id": row.get("id", str(index)), "prediction": prediction, "correct": correct}
        except Exception as error:
            # Operational failures remain in the accuracy denominator and are reported separately.
            result = {"id": row.get("id", str(index)), "error": f"{type(error).__name__}: {error}",
                      "correct": False}
        result.update(calls=pool.calls, wall_seconds=time.monotonic() - started)
        results.append(result)
        with (output_dir / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
    failures = sum("error" in row for row in results)
    summary = {"examples": len(results), "correct": sum(r["correct"] for r in results),
               "accuracy": sum(r["correct"] for r in results) / len(results),
               "operational_failures": failures, "calls": sum(r["calls"] for r in results),
               "wall_seconds": sum(r["wall_seconds"] for r in results)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

