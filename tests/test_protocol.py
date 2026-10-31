import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from grounding.client import BudgetExceeded, ModelPool
from grounding.config import load_config
from grounding.controller_diagnostic import ground, parse_action, validate_point
from grounding.evaluation import score_point
from grounding.tools import PythonSession


def config():
    return {"models": [{"alias": "qwen", "port": 8001}, {"alias": "lfm", "port": 8003}]}


@pytest.fixture
def screenshot(tmp_path):
    path = tmp_path / "screen.png"
    Image.new("RGB", (400, 200), "white").save(path)
    return path


def test_failed_model_request_consumes_budget(monkeypatch):
    pool = ModelPool(config(), max_calls=1)

    def fail(*args, **kwargs):
        raise httpx.ConnectError("server unavailable")

    monkeypatch.setattr(httpx, "post", fail)
    with pytest.raises(httpx.ConnectError):
        pool.ask("qwen", "hello")
    with pytest.raises(BudgetExceeded):
        pool.ask("lfm", "hello")
    assert pool.calls == 1
    assert "ConnectError" in pool.records[0]["error"]


def test_trace_preserves_text_but_omits_image_bytes(monkeypatch, screenshot, tmp_path):
    def post(url, *, json, timeout):
        assert json["max_tokens"] == 32
        assert json["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"content": "[0.5,0.5]"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10},
        })
    monkeypatch.setattr(httpx, "post", post)
    trace = tmp_path / "trace.jsonl"
    pool = ModelPool(config(), max_output_tokens=32, trace_path=trace)
    assert pool.ask("qwen", "Find Save", image=screenshot) == "[0.5,0.5]"
    text = trace.read_text()
    assert "Find Save" in text and "<image omitted>" in text
    assert "base64" not in text


def test_python_persists_state_routes_model_calls_and_displays_image(screenshot):
    calls = []

    def ask(model, prompt, **kwargs):
        calls.append((model, prompt, kwargs))
        return "the Save button"

    pool = SimpleNamespace(models={"qwen": {}, "lfm": {}}, ask=ask)
    with PythonSession(screenshot, "Click Save", pool) as python:
        result = python.execute("x = 42\nprint(image.size, x)\ny = call_model('lfm', 'Read', image=image)\nshow_image(image)")
        assert result["error"] is None
        assert "(400, 200) 42" in result["stdout"]
        assert len(result["images"]) == 1
        assert calls[0][0] == "lfm"
        assert calls[0][2]["encoded_image"].startswith("data:image/png;base64,")
        assert "42 the Save button" in python.execute("print(x, y)")["stdout"]


def test_python_does_not_inherit_credentials(screenshot, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-value-never-forward")
    pool = SimpleNamespace(models={})
    with PythonSession(screenshot, "test", pool) as python:
        result = python.execute("import os\nprint(os.getenv('MODAL_TOKEN_SECRET'), os.getenv('CUDA_VISIBLE_DEVICES'))")
        assert result["stdout"].strip() == "None"


def test_python_timeout_stops_interpreter(screenshot):
    with PythonSession(screenshot, "test", SimpleNamespace(models={}), timeout=0.2) as python:
        with pytest.raises(TimeoutError):
            python.execute("while True: pass")
        assert python.process.poll() is not None


def test_controller_executes_python_before_final(screenshot):
    class FakePool:
        models = {"qwen": {}, "lfm": {}}
        max_calls = 4
        calls = 0

        def chat(self, alias, messages):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"action": "python", "code": "print(image.size)"})
            assert "(400, 200)" in messages[-1]["content"][0]["text"]
            return '{"action":"final","point":[0.5,0.5]}'

    result = ground(screenshot, "Click center", FakePool())
    assert result["status"] == "completed" and result["point"] == [0.5, 0.5]
    assert len(result["steps"]) == 1


def test_invalid_points_cannot_be_scored_as_valid():
    for point in ([float("nan"), 0.5], [float("inf"), 0], [-1, 0.5], [True, 0.5], [0.5]):
        with pytest.raises(ValueError):
            validate_point(point)
    assert score_point([0.5, 0.5], [190, 90, 210, 110], 400, 200)
    assert not score_point([0.1, 0.1], [190, 90, 210, 110], 400, 200)


def test_action_parser_handles_fences_and_thinking():
    assert parse_action('<think>not JSON</think>```json\n{"action":"final","point":null}\n```')["point"] is None


def test_pinned_pool_preserves_headroom():
    cfg = load_config(Path(__file__).parents[1] / "models.json")
    assert len(cfg["models"]) == 3
    assert sum(m["gpu_memory_utilization"] for m in cfg["models"]) == 0.75


def test_python_ground_model_passes_original_frame(screenshot):
    def localize(alias, instruction, **kwargs):
        assert alias == "qwen" and instruction == "Click Save"
        assert kwargs["frame"]["original_size"] == [400, 200]
        return {"point_original_norm01": [0.75, 0.5], "coordinate_space": "norm1000"}

    pool = SimpleNamespace(models={"qwen": {}}, ground=localize)
    with PythonSession(screenshot, "Click Save", pool) as python:
        result = python.execute("print(ground_model('qwen', instruction)['point_original_norm01'])")
        assert result["error"] is None
        assert "[0.75, 0.5]" in result["stdout"]


def test_resolution_and_json_constraints_reach_vllm(monkeypatch):
    def post(url, *, json, timeout):
        assert json['mm_processor_kwargs']=={'max_pixels':4194304}
        assert json['structured_outputs']['json']['type']=='object'
        assert json['temperature']==0.7 and json['top_p']==0.9
        return httpx.Response(200,request=httpx.Request('POST',url),json={
            'choices':[{'message':{'content':'{"action":"final","point":[500,500]}'}}]})
    monkeypatch.setattr(httpx,'post',post)
    pool=ModelPool(config())
    pool.temperature,pool.top_p=0.7,0.9
    pool.chat('qwen',[{'role':'user','content':'test'}],processor_kwargs={'max_pixels':4194304},
              response_schema={'type':'object'})
    assert pool.calls==1

