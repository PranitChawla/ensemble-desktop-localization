import json

import httpx
import pytest
from PIL import Image

from grounding.client import BudgetExceeded, ModelPool
from grounding.config import load_config


def test_failed_http_request_consumes_shared_budget(monkeypatch):
    pool = ModelPool(load_config(), max_calls=1)

    def fail(*args, **kwargs):
        raise httpx.ConnectError("Unavailable")

    monkeypatch.setattr(httpx, "post", fail)
    with pytest.raises(httpx.ConnectError):
        pool.ask_vlm("qwen", "Classify this")
    with pytest.raises(BudgetExceeded):
        pool.ask_vlm("mai", "Verify this")
    assert pool.calls == 1 and "error" in pool.records[0]


def test_general_questions_and_python_images_preserve_constraints(monkeypatch, tmp_path):
    def post(url, *, json, timeout):
        assert len(json["messages"][0]["content"]) == 3
        assert json["structured_outputs"]["json"]["properties"]["kind"]["enum"] == ["icon", "text"]
        assert json["temperature"] == 0.7 and json["top_p"] == 0.9
        assert json["max_tokens"] == 64
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"kind":"icon"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", post)
    trace = tmp_path / "trace.jsonl"
    pool = ModelPool(load_config(), trace_path=trace)
    result = pool.ask_vlm(
        "qwen",
        "Classify",
        images=[Image.new("RGB", (20, 20)), Image.new("RGB", (10, 10))],
        response_schema={"type": "object", "properties": {"kind": {"enum": ["icon", "text"]}}},
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
    )
    assert json.loads(result) == {"kind": "icon"}
    assert "<image omitted>" in trace.read_text() and "base64" not in trace.read_text()


def test_output_and_image_budgets_are_fixed(monkeypatch):
    def post(url, *, json, timeout):
        assert json["max_tokens"] == 512
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"choices": [{"message": {"content": "OK"}}]}
        )

    monkeypatch.setattr(httpx, "post", post)
    pool = ModelPool(load_config())
    assert pool.ask_vlm("qwen", "Read", max_tokens=10000) == "OK"
    with pytest.raises(ValueError, match="input budget"):
        pool.ask_vlm("qwen", "Read", processor_kwargs={"max_pixels": 100000000})
    with pytest.raises(ValueError, match="Too many images"):
        pool.ask_vlm("qwen", "Read", images=[Image.new("RGB", (10, 10))] * 4)
    assert pool.calls == 1


def test_three_model_memory_fractions_reserve_headroom():
    config = load_config()
    assert len(config["models"]) == 3
    assert sum(m["gpu_memory_utilization"] for m in config["models"]) == 0.75
