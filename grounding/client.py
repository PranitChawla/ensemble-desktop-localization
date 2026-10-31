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
    """Authoritative shared budget for all submission-initiated VLM requests."""

    def __init__(
        self,
        config: dict,
        *,
        max_calls: int = 6,
        max_output_tokens: int = 512,
        timeout: float = 45,
        trace_path: str | Path | None = None,
    ):
        self.models = {m["alias"]: m for m in config["models"]}
        self.max_calls = max_calls
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.trace_path = Path(trace_path) if trace_path else None
        self.temperature, self.top_p = 0.0, 1.0
        self.max_images = 3
        self.max_image_bytes = 24000000
        self.calls = 0
        self.records: list[dict] = []
        self._lock = threading.RLock()

    def chat(
        self,
        alias: str,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        processor_kwargs: dict | None = None,
        response_schema: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        # Serialize across all servers, not just within each vLLM instance.
        with self._lock:
            if alias not in self.models:
                raise ValueError(f"Unknown model {alias!r}; available: {list(self.models)}")
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"Total model-call budget exhausted ({self.max_calls})")
            model = self.models[alias]
            images = [
                part["image_url"]["url"]
                for message in messages
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "image_url"
            ]
            if len(images) > self.max_images:
                raise ValueError("Too many images in a VLM request")
            for encoded in images:
                if not encoded.startswith("data:image/"):
                    raise ValueError("Use local image data; remote image URLs are not allowed")
                if len(encoded) > self.max_image_bytes * 4 // 3 + 200:
                    raise ValueError("Encoded image exceeds the task image-byte limit")
            request_temperature = self.temperature if temperature is None else temperature
            request_top_p = self.top_p if top_p is None else top_p
            payload = {
                "model": alias,
                "messages": messages,
                "temperature": request_temperature,
                "top_p": request_top_p,
                "max_tokens": min(max_tokens or self.max_output_tokens, self.max_output_tokens),
            }
            if model.get("chat_template_kwargs"):
                payload["chat_template_kwargs"] = model["chat_template_kwargs"]
            if processor_kwargs is None:
                processor_kwargs = model.get("request_processor_kwargs")
            if processor_kwargs is not None:
                if set(processor_kwargs) - {"max_pixels"}:
                    raise ValueError("Only max_pixels may be overridden by submissions")
                maximum = model.get("max_request_pixels", model.get("processor_kwargs", {}).get("max_pixels"))
                pixels = processor_kwargs.get("max_pixels")
                if pixels is not None and (
                    isinstance(pixels, bool)
                    or not isinstance(pixels, int)
                    or pixels <= 0
                    or (maximum is not None and pixels > maximum)
                ):
                    raise ValueError("max_pixels exceeds the fixed model input budget")
                payload["mm_processor_kwargs"] = processor_kwargs
            if response_schema is not None:
                payload["structured_outputs"] = {"json": response_schema}
            self.calls += 1  # Every actual request counts, including HTTP failures; no hidden retries.
            started = time.monotonic()
            record = {
                "call": self.calls,
                "model": alias,
                "messages": messages,
                "processor_kwargs": processor_kwargs,
                "response_schema": response_schema,
                "temperature": request_temperature,
                "top_p": request_top_p,
            }
            try:
                response = httpx.post(
                    f"http://127.0.0.1:{model['port']}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                message = result["choices"][0]["message"]
                text = message.get("content") or ""
                record.update(
                    response=text,
                    usage=result.get("usage", {}),
                    finish_reason=result["choices"][0].get("finish_reason"),
                )
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

    def ask(
        self,
        alias: str,
        prompt: str,
        *,
        image: str | Path | None = None,
        encoded_image: str | None = None,
        processor_kwargs: dict | None = None,
    ) -> str:
        content = [{"type": "text", "text": prompt}]
        if image is not None or encoded_image is not None:
            content.append({"type": "image_url", "image_url": {"url": encoded_image or image_url(image)}})
        return self.chat(alias, [{"role": "user", "content": content}], processor_kwargs=processor_kwargs)

    def ask_vlm(
        self,
        alias: str,
        question: str,
        *,
        image=None,
        images=None,
        response_schema=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
        processor_kwargs=None,
    ) -> str:
        """General visual question answering; the caller interprets arbitrary text/JSON.

        Images may be paths or PIL images, including Python-created representations.
        No coordinate parser or task taxonomy is imposed. All calls share the budget.
        """
        if image is not None and images is not None:
            raise ValueError("Provide image or images, not both")
        supplied = list(images) if images is not None else ([] if image is None else [image])
        content = [{"type": "text", "text": str(question)}]
        for value in supplied:
            encoded = encode_image(value) if hasattr(value, "save") else image_url(value)
            content.append({"type": "image_url", "image_url": {"url": encoded}})
        return self.chat(
            alias,
            [{"role": "user", "content": content}],
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            processor_kwargs=processor_kwargs,
        )

    def ground(
        self,
        alias: str,
        instruction: str,
        *,
        image=None,
        encoded_image=None,
        frame=None,
        processor_kwargs=None,
    ):
        from grounding.adapters import ground

        return ground(
            self,
            alias,
            instruction,
            encoded_image=encoded_image or image_url(image),
            frame=frame,
            processor_kwargs=processor_kwargs,
        )
