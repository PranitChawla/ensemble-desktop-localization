"""Label-owning evaluator; submission code runs in a separate per-example process."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from grounding.client import ModelPool
from grounding.coordinates import finite_values, point_to_input_norm01
from grounding.runtime import CandidateRunner
from grounding.task import load_task


def score_point(point, box, width, height):
    if point is None:
        return box is None
    try:
        point = point_to_input_norm01(point, "norm01", (width, height))
    except ValueError:
        return False
    if box is None:
        return False
    x, y = point[0] * width, point[1] * height
    return bool(box[0] <= x <= box[2] and box[1] <= y <= box[3])


def read_manifest(path, limit=None):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("Empty evaluation manifest")
    identities = set()
    for index, row in enumerate(rows):
        identity = str(row.get("id", index))
        if identity in identities:
            raise ValueError("Duplicate manifest identifier")
        identities.add(identity)
        if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
            raise ValueError("Each example requires a non-empty instruction")
        image = (path.parent / row["image"]).resolve()
        with Image.open(image) as source:
            width, height = source.size
        if "bbox_xyxy" not in row:
            raise ValueError("Manifest requires bbox_xyxy; null means a refusal target")
        if row["bbox_xyxy"] is not None:
            x0, y0, x1, y1 = finite_values(row["bbox_xyxy"], 4)
            if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
                raise ValueError("Ground-truth box must use bounded original-image pixels")
        yield dict(row, id=identity, path=image, width=width, height=height)


def evaluate(manifest, candidate, config, output_dir, *, limit=None, task_path=None, max_calls=None):
    task = load_task(task_path) if task_path else load_task()
    budget = task["budget"]
    if max_calls is not None and not 1 <= max_calls <= budget["max_model_calls"]:
        raise ValueError("Call limit cannot exceed the task budget")
    rows = list(read_manifest(manifest, limit))  # Validate all selected rows before making any model request.
    candidate = Path(candidate).resolve()
    if not candidate.is_file():
        raise ValueError("Submission file does not exist")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "task": task,
        "models": config,
        "models_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "planned_examples": len(rows),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    for index, row in enumerate(rows):
        pool = ModelPool(
            config,
            max_calls=max_calls or budget["max_model_calls"],
            max_output_tokens=budget["max_output_tokens_per_call"],
            timeout=budget["request_timeout_seconds"],
            trace_path=output / f"case-{index:05d}.jsonl",
        )
        pool.max_images = budget["max_images_per_request"]
        pool.max_image_bytes = budget["max_image_bytes"]
        start = time.monotonic()
        result = {"id": row["id"], "correct": False}
        try:
            event = CandidateRunner(pool, timeout=budget["case_timeout_seconds"]).run(
                candidate, row["path"], row["instruction"]
            )
            result["stdout"] = event.get("stdout", "")
            if event.get("error"):
                raise RuntimeError(event["error"])
            prediction = event.get("prediction")
            if not isinstance(prediction, dict) or "point" not in prediction:
                raise ValueError("Submission must return a JSON object with a point field")
            point = prediction["point"]
            if point is not None:
                point = point_to_input_norm01(point, "norm01", (row["width"], row["height"]))
            result.update(
                status="completed",
                prediction=prediction,
                point=point,
                correct=score_point(point, row["bbox_xyxy"], row["width"], row["height"]),
            )
        except Exception as error:
            result.update(
                status="timeout" if isinstance(error, TimeoutError) else "error",
                error=f"{type(error).__name__}: {error}",
            )
        result.update(
            calls=pool.calls,
            wall_seconds=time.monotonic() - start,
            prompt_tokens=sum(r.get("usage", {}).get("prompt_tokens", 0) for r in pool.records),
            output_tokens=sum(r.get("usage", {}).get("completion_tokens", 0) for r in pool.records),
            request_errors=sum("error" in r for r in pool.records),
        )
        results.append(result)
        with (output / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
    summary = {
        "examples": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "operational_failures": sum(r["status"] != "completed" for r in results),
        "calls": sum(r["calls"] for r in results),
        "prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "output_tokens": sum(r["output_tokens"] for r in results),
        "wall_seconds": sum(r["wall_seconds"] for r in results),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
