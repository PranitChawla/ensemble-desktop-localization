"""Execute agent-authored submission code. VLM responses are never executed."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import threading
import traceback
import types
from pathlib import Path

package = types.ModuleType("grounding")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules["grounding"] = package
from grounding.client import encode_image, image_url  # noqa: E402
from grounding.coordinates import ImageFrame  # noqa: E402

PROTOCOL_IN, PROTOCOL_OUT = sys.stdin, sys.stdout


def send(value):
    PROTOCOL_OUT.write(json.dumps(value, allow_nan=False) + "\n")
    PROTOCOL_OUT.flush()


class BoundedOutput(io.StringIO):
    def write(self, text):
        super().write(text[: max(0, 16000 - self.tell())])
        return len(text)


class VLMProxy:
    def __init__(self, models, max_calls):
        self.models = models
        self.calls = 0
        self.max_calls = max_calls
        self.lock = threading.RLock()

    def _request(self, method, alias, **kwargs):
        with self.lock:
            send({"type": "model_call", "method": method, "alias": alias, "kwargs": kwargs})
            response = json.loads(PROTOCOL_IN.readline())
            self.calls = response.get("calls", self.calls)
            if response.get("error"):
                raise RuntimeError(response["error"])
            return response["result"]

    def chat(self, alias, messages, **kwargs):
        return self._request("chat", alias, messages=messages, **kwargs)

    def ask_vlm(self, alias, question, *, image=None, images=None, **kwargs):
        if image is not None and images is not None:
            raise ValueError("Provide image or images, not both")
        supplied = list(images) if images is not None else ([] if image is None else [image])
        content = [{"type": "text", "text": str(question)}]
        for value in supplied:
            encoded = encode_image(value) if hasattr(value, "save") else image_url(value)
            content.append({"type": "image_url", "image_url": {"url": encoded}})
        return self.chat(alias, [{"role": "user", "content": content}], **kwargs)

    def ask(self, alias, prompt, *, image=None, encoded_image=None, **kwargs):
        if encoded_image is not None:
            content = [
                {"type": "text", "text": str(prompt)},
                {"type": "image_url", "image_url": {"url": encoded_image}},
            ]
            return self.chat(alias, [{"role": "user", "content": content}], **kwargs)
        return self.ask_vlm(alias, prompt, image=image, **kwargs)

    def ground(self, alias, instruction, *, image=None, encoded_image=None, frame=None, **kwargs):
        if isinstance(frame, ImageFrame):
            frame = frame.to_dict()
        if encoded_image is None:
            encoded_image = encode_image(image) if hasattr(image, "save") else image_url(image)
        return self._request(
            "ground", alias, instruction=instruction, encoded_image=encoded_image, frame=frame, **kwargs
        )


def main():
    init = json.loads(PROTOCOL_IN.readline())
    pool = VLMProxy(init["models"], init["max_calls"])
    output = BoundedOutput()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            spec = importlib.util.spec_from_file_location("candidate_submission", init["candidate"])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            result = module.ground(init["screenshot"], init["instruction"], pool)
            encoded = json.dumps(result, allow_nan=False)
            if len(encoded.encode()) > 64000:
                raise ValueError("Submission result exceeds the 64 KB JSON limit")
        send({"type": "result", "prediction": result, "stdout": output.getvalue()})
    except BaseException:
        send({"type": "result", "error": traceback.format_exc()[-4000:], "stdout": output.getvalue()})


if __name__ == "__main__":
    main()
