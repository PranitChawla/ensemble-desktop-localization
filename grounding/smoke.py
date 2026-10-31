from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from grounding.client import ModelPool
from grounding.coordinates import ImageFrame
from grounding.servers import gpu_memory
from grounding.tools import PythonSession


def make_screenshot(path: str | Path):
    image = Image.new("RGB", (1920, 1080), "#e8edf3")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        title, button = ImageFont.truetype(font_path, 36), ImageFont.truetype(font_path, 28)
    except OSError:
        title = button = ImageFont.load_default(size=28)
    draw.rectangle((0, 0, 1920, 90), fill="#263445")
    draw.text((40, 24), "Document Editor", font=title, fill="white")
    draw.rectangle((160, 180, 1760, 880), fill="white", outline="#9aa8b7", width=3)
    draw.text((210, 230), "Project settings", font=title, fill="#263445")
    draw.text((210, 320), "Name: Initial grounding plumbing test", font=button, fill="#263445")
    draw.rounded_rectangle((1380, 760, 1670, 840), radius=10, fill="#1765c1")
    draw.text((1460, 782), "Save", font=button, fill="white")
    draw.rounded_rectangle((1040, 760, 1330, 840), radius=10, fill="#d8dee6")
    draw.text((1125, 782), "Cancel", font=button, fill="#263445")
    image.save(path)
    return {"instruction": "Click the Save button", "bbox_xyxy": [1380, 760, 1670, 840]}


def smoke(config: dict, output_dir: str | Path, *, long_context=True) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot = output_dir / "synthetic-desktop.png"
    target = make_screenshot(screenshot)
    pool = ModelPool(config, max_calls=20, trace_path=output_dir / "model-calls.jsonl")
    report = {"kind": "synthetic_plumbing_not_benchmark", "models": [], "memory": []}
    for model in config["models"]:
        started = time.monotonic()
        grounding = pool.ground(model["alias"], target["instruction"],
                               image=screenshot, frame=ImageFrame.full((1920, 1080)))
        text = grounding["raw_response"]
        if not text.strip():
            raise RuntimeError(f"{model['alias']} returned an empty screenshot response")
        item = {"alias": model["alias"], "vision_response": text, "vision_seconds": time.monotonic() - started,
                "vision_usage": pool.records[-1].get("usage", {}), "grounding": grounding}
        if long_context:
            # Real model requests exercise the KV cache, rather than just checking /health.
            # Actual prompt_tokens are recorded; this is a ~8K-token probe, not a 16K capacity proof.
            filler = "\n".join(f"Tool observation {i}: dimensions=1920x1080; working memory entry." for i in range(450))
            text = pool.ask(model["alias"], filler + "\nNow answer only READY.")
            item.update(context_response=text, context_usage=pool.records[-1].get("usage", {}))
        report["models"].append(item)
        report["memory"].append({"stage": f"requests_{model['alias']}", **gpu_memory()})
        print(json.dumps(item), flush=True)
    with PythonSession(screenshot, target["instruction"], pool) as python:
        result = python.execute(
            "print('persistent', image.size, 6 * 7)\n"
            "answer = call_model('lfm', 'Read the labels on the two buttons.', image=image)\n"
            "print(answer)\nshow_image(image.resize((960, 540)))"
        )
        second = python.execute("print('state persisted:', answer[:80])")
        if result["error"] or second["error"] or len(result["images"]) != 1:
            raise RuntimeError(f"Python tool plumbing failed: {result['error']} / {second['error']}")
        report["python_tool"] = {"stdout": result["stdout"], "error": result["error"],
                                  "displayed_images": len(result["images"]), "second_stdout": second["stdout"]}
    import submission
    agent_pool = ModelPool(config, max_calls=12, trace_path=output_dir / "agent-calls.jsonl")
    report["agent"] = submission.ground(str(screenshot), target["instruction"], agent_pool)
    if report["agent"]["status"] != "completed":
        raise RuntimeError(f"Agent did not produce a final action: {report['agent']}")
    from grounding.evaluation import score_point
    report["agent"]["synthetic_target_hit"] = (
        report["agent"]["status"] == "completed" and
        score_point(report["agent"]["point"], target["bbox_xyxy"], 1920, 1080)
    )
    report["memory"].append({"stage": "after_agent", **gpu_memory()})
    report["call_count"] = pool.calls + agent_pool.calls
    (output_dir / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

