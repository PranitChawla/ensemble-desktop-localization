"""Checkpoint-specific inference protocols, audited against publisher code."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image

from grounding.coordinates import ImageFrame, normalize_grounding, to_original_point


def smart_size(size, *, factor=28, min_pixels=12544, max_pixels=2116800):
    """Qwen2-VL smart-resize geometry: patch/merge aligned and bounded by area."""
    width, height = size
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return w, h


def ground(pool, alias, instruction, *, encoded_image, frame=None, processor_kwargs=None):
    import base64
    import io

    from grounding.client import encode_image
    from grounding.coordinates import grounding_prompt

    with Image.open(io.BytesIO(base64.b64decode(encoded_image.split(",", 1)[1]))) as decoded:
        supplied = decoded.convert("RGB")
    if frame is not None:
        checked = ImageFrame(**frame) if isinstance(frame, dict) else frame
        if tuple(checked.image_size) != tuple(supplied.size):
            raise ValueError("Frame dimensions do not match the image actually supplied")
    convention = pool.models[alias]["grounding"]
    adapter = convention.get("adapter")
    units = convention["request_space"]
    size = supplied.size
    if adapter == "jedi":
        size = smart_size(
            supplied.size,
            max_pixels=pool.models[alias].get("processor_kwargs", {}).get("max_pixels", 2116800),
        )
        supplied = supplied.resize(size, Image.Resampling.BICUBIC)
        schema = json.loads((Path(__file__).parent / "templates/jedi_computer_use.json").read_text())
        schema["description"] = schema["description"].format(width=size[0], height=size[1])
        tools = json.dumps({"type": "function", "function": schema}, ensure_ascii=False)
        system = (
            "You are a helpful assistant.\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(supplied)}},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        raw = pool.chat(alias, messages, processor_kwargs=processor_kwargs)
    elif adapter == "venus":
        prompt = (
            "Output the center point of the position corresponding to the following instruction: \n"
            f"{instruction.rstrip('.')}. \n\n"
            "The output should just be the coordinates of a point, in the format [x,y]. "
            "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
            "the output should be [-1,-1]."
        )
        raw = pool.chat(
            alias,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encoded_image}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            processor_kwargs=processor_kwargs,
        )
    elif adapter == "mai":
        prompt = (Path(__file__).parent / "templates/mai_grounding.txt").read_text().strip()
        raw = pool.chat(
            alias,
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction + "\n"},
                        {"type": "image_url", "image_url": {"url": encoded_image}},
                    ],
                },
            ],
            processor_kwargs=processor_kwargs,
        )
    else:
        raw = pool.ask(
            alias,
            grounding_prompt(instruction, units, size),
            encoded_image=encoded_image,
            processor_kwargs=processor_kwargs,
        )
    parse_text = raw
    if adapter == "mai":
        answer = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if not answer:
            raise ValueError("MAI response lacks its documented answer tag")
        parse_text = answer.group(1)
    if adapter == "venus" and raw.strip().replace(" ", "") == "[-1,-1]":
        result = normalize_grounding("null", units, size)
    else:
        result = normalize_grounding(parse_text, units, size)
    # Explicit resize preserves relative positions. Compose with the caller's frame.
    point = result["point_input_norm01"]
    result["point_original_norm01"] = (
        to_original_point(point, "norm01", frame) if point is not None and frame else None
    )
    result.update(
        raw_response=raw,
        format_status=convention["status"],
        native_space=convention["native_space"],
        frame=frame.to_dict() if isinstance(frame, ImageFrame) else frame,
        supplied_image_size=list(decoded.size),
        processor_image_size=list(size) if adapter == "jedi" else None,
    )
    return result
