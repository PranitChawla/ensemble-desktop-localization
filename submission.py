"""Editable research surface: open-ended Python-tool-assisted GUI grounding.

No crop pipeline or fixed ensemble is imposed. The controller chooses its Python
programs, model calls, intermediate representations, and final location.
"""
from __future__ import annotations

import json
import math
import re

from grounding.client import image_url
from grounding.tools import PythonSession

SYSTEM_PROMPT = """You ground desktop GUI instructions. You have a persistent Python tool.
Return exactly one JSON object per turn, without markdown:
{"action":"python","code":"... arbitrary Python code ..."}
or {"action":"final","point":[x,y]} with x,y normalized to [0,1] in the ORIGINAL screenshot,
or {"action":"final","point":null} if the requested target does not exist.

Python globals: image (PIL original screenshot), screenshot_path, instruction,
models (available aliases), np, Image, ImageDraw, ImageFont.
call_model(alias, prompt, image=None) calls another frozen VLM and returns text.
Pass a PIL image or file path for image. show_image(image_or_path) displays an image
to you on the next turn. Variables persist. Imports and arbitrary Python are allowed.
Use computation, other models, or visual inspection whenever useful. No particular
image manipulation or ensembling procedure is required. There is no GUI interaction:
the screenshot is static. Do not assume a proposed coordinate is correct without evidence.
All controller and helper calls share a small total call budget. Be economical.
"""


def parse_action(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character == "{":
            try:
                action, _ = decoder.raw_decode(cleaned[index:])
                if isinstance(action, dict) and action.get("action") in {"python", "final"}:
                    return action
            except ValueError:
                continue
    raise ValueError("Expected one JSON python/final action")


def validate_point(point):
    if point is None:
        return None
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError("point must be [x,y] or null")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1 for v in point):
        raise ValueError("Coordinates must be finite normalized values in [0,1]")
    return [float(v) for v in point]


def ground(screenshot, instruction, pool, *, controller="qwen", max_steps=6, tool_timeout=180):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Instruction: {instruction}\nAvailable models: {list(pool.models)}"},
            {"type": "image_url", "image_url": {"url": image_url(screenshot)}},
        ]},
    ]
    steps = []
    with PythonSession(screenshot, instruction, pool, timeout=tool_timeout) as python:
        for _ in range(max_steps):
            response = pool.chat(controller, messages)
            messages.append({"role": "assistant", "content": response})
            try:
                action = parse_action(response)
                if action["action"] == "final":
                    point = validate_point(action["point"])
                    return {"status": "completed", "point": point, "steps": steps, "calls": pool.calls}
                if not isinstance(action.get("code"), str):
                    raise ValueError("Python action requires a code string")
            except (ValueError, KeyError) as error:
                messages.append({"role": "user", "content": f"Invalid action: {error}. Correct the JSON."})
                steps.append({"error": str(error), "response": response})
                continue
            result = python.execute(action["code"])
            steps.append({"code": action["code"], "stdout": result["stdout"], "error": result["error"]})
            content = [{"type": "text", "text": json.dumps({
                "stdout": result["stdout"], "error": result["error"],
                "calls_remaining": pool.max_calls - pool.calls,
            })}]
            content.extend({"type": "image_url", "image_url": {"url": value}}
                           for value in result.get("images", []))
            # Keep only the newest tool displays; retain the original screenshot and text history.
            for prior in messages[2:]:
                if isinstance(prior.get("content"), list):
                    prior["content"] = [p for p in prior["content"] if p["type"] != "image_url"]
            messages.append({"role": "user", "content": content})
    return {"status": "step_limit", "point": None, "steps": steps, "calls": pool.calls}

