"""Persistent, arbitrary Python interpreter. Protocol travels on the original stdio.

This is a research tool, NOT a hostile-code security sandbox. Run in a disposable
container with no secrets. The parent owns model calls, accounting, and timeouts.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROTOCOL_IN, PROTOCOL_OUT = sys.stdin, sys.stdout


def send(value):
    PROTOCOL_OUT.write(json.dumps(value) + "\n")
    PROTOCOL_OUT.flush()


def encode_image(value):
    image = value if isinstance(value, Image.Image) else Image.open(value)
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    if output.tell() > 12_000_000:
        raise ValueError("Tool images must be at most 12 MB")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class BoundedOutput(io.StringIO):
    def write(self, text):
        remaining = max(0, 16000 - self.tell())
        super().write(text[:remaining])
        return len(text)


def main():
    init = json.loads(PROTOCOL_IN.readline())
    rendered_images = []

    def call_model(model, prompt, image=None):
        send({"type": "model_call", "model": model, "prompt": str(prompt),
              "encoded_image": encode_image(image) if image is not None else None})
        reply = json.loads(PROTOCOL_IN.readline())
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply["text"]

    def show_image(image):
        if len(rendered_images) >= 2:
            raise ValueError("At most two displayed images per Python execution")
        rendered_images.append(encode_image(image))

    namespace = {
        "__name__": "__grounding_tool__", "np": np, "Image": Image,
        "ImageDraw": ImageDraw, "ImageFont": ImageFont,
        "screenshot_path": init["screenshot_path"],
        "image": Image.open(init["screenshot_path"]).convert("RGB"),
        "instruction": init["instruction"], "models": init["models"],
        "call_model": call_model, "show_image": show_image,
    }
    send({"type": "ready"})
    for line in PROTOCOL_IN:
        request = json.loads(line)
        rendered_images.clear()
        output = BoundedOutput()
        error = None
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(request["code"], "<python-tool>", "exec"), namespace)
        except BaseException:
            error = traceback.format_exc()[-4000:]
        send({"type": "result", "stdout": output.getvalue(), "error": error,
              "images": list(rendered_images)})


if __name__ == "__main__":
    main()

