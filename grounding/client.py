from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path

import httpx


class BudgetExceeded(RuntimeError):
    pass


def encode_image(image) -> str:
    import io

    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def image_url(path: str | Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        return encode_image(image)


class ModelPool:
    """One authoritative call budget for controller and Python-initiated requests."""

    def __init__(self, config: dict, *, max_calls: int = 12, max_output_tokens: int = 1024,
                 timeout: float = 180, trace_path: str | Path | None = None):
        self.models = {m["alias"]: m for m in config["models"]}
        self.max_calls = max_calls
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.trace_path = Path(trace_path) if trace_path else None
        self.temperature, self.top_p = 0.0, 1.0
        self.calls = 0
        self.records: list[dict] = []
        self._lock = threading.RLock()

    def chat(self, alias: str, messages: list[dict], *, max_tokens: int | None = None,
             processor_kwargs: dict | None = None, response_schema: dict | None = None) -> str:
        # Serialize across all servers, not just within each vLLM instance.
        with self._lock:
            if alias not in self.models:
                raise ValueError(f"Unknown model {alias!r}; available: {list(self.models)}")
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"Total model-call budget exhausted ({self.max_calls})")
            self.calls += 1  # Failed requests count; no silent retries or resampling.
            model = self.models[alias]
            payload = {
                "model": alias, "messages": messages, "temperature": self.temperature, "top_p": self.top_p,
                "max_tokens": min(max_tokens or self.max_output_tokens, self.max_output_tokens),
            }
            if model.get("chat_template_kwargs"):
                payload["chat_template_kwargs"] = model["chat_template_kwargs"]
            if processor_kwargs is None:
                processor_kwargs = model.get("request_processor_kwargs")
            if processor_kwargs is not None:
                payload["mm_processor_kwargs"] = processor_kwargs
            if response_schema is not None:
                payload["structured_outputs"] = {"json": response_schema}
            started = time.monotonic()
            record = {"call": self.calls, "model": alias, "messages": messages,
                      "processor_kwargs": processor_kwargs, "response_schema": response_schema,
                      "temperature": self.temperature, "top_p": self.top_p}
            try:
                response = httpx.post(
                    f"http://127.0.0.1:{model['port']}/v1/chat/completions",
                    json=payload, timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                message = result["choices"][0]["message"]
                text = message.get("content") or ""
                record.update(response=text, usage=result.get("usage", {}),
                              finish_reason=result["choices"][0].get("finish_reason"))
                return text
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {str(error)[:1200]}"
                if isinstance(error, httpx.HTTPStatusError):
                    record["server_error"] = error.response.text[:4000]
                raise
            finally:
                record["wall_seconds"] = time.monotonic() - started
                self.records.append(record)
                if self.trace_path:
                    self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                    # Exclude base64 blobs from durable traces, while retaining text and model outputs.
                    safe = json.loads(json.dumps(record))
                    for msg in safe.get("messages", []):
                        if isinstance(msg.get("content"), list):
                            for part in msg["content"]:
                                if part.get("type") == "image_url":
                                    part["image_url"] = {"url": "<image omitted>"}
                    with self.trace_path.open("a") as stream:
                        stream.write(json.dumps(safe) + "\n")

    def ask(self, alias: str, prompt: str, *, image: str | Path | None = None,
            encoded_image: str | None = None, processor_kwargs: dict | None = None) -> str:
        content = [{"type": "text", "text": prompt}]
        if image is not None or encoded_image is not None:
            content.append({"type": "image_url", "image_url": {"url": encoded_image or image_url(image)}})
        return self.chat(alias, [{"role": "user", "content": content}], processor_kwargs=processor_kwargs)

    def ground(self, alias: str, instruction: str, *, image=None, encoded_image=None, frame=None, processor_kwargs=None):
        from grounding.adapters import ground

        return ground(self,alias,instruction,encoded_image=encoded_image or image_url(image),frame=frame,
                      processor_kwargs=processor_kwargs)
