import json
import time

import httpx
import pytest
from PIL import Image

from grounding.client import ModelPool
from grounding.config import load_config
from grounding.evaluation import evaluate
from grounding.runtime import CandidateRunner


def files(tmp_path, source):
    image = tmp_path / "original-target-name.png"
    Image.new("RGB", (400, 200)).save(image)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(source)
    return candidate, image


def test_candidate_classifies_then_grounds_through_parent_rpc(tmp_path, monkeypatch):
    candidate, image = files(
        tmp_path,
        """import json\nfrom PIL import Image\nfrom grounding.coordinates import ImageFrame\ndef ground(screenshot,instruction,pool):\n assert 'port' not in pool.models['qwen']\n kind=json.loads(pool.ask_vlm('qwen','Classify',image=screenshot))['kind']\n with Image.open(screenshot) as img:frame=ImageFrame.full(img.size)\n answer=pool.ground('venus',instruction,image=screenshot,frame=frame)\n print(kind,pool.calls)\n return {'point':answer['point_original_norm01']}\n""",
    )

    def post(url, *, json, timeout):
        text = '{"kind":"icon"}' if json["model"] == "qwen" else "[500,500]"
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"choices": [{"message": {"content": text}}]}
        )

    monkeypatch.setattr(httpx, "post", post)
    pool = ModelPool(load_config())
    event = CandidateRunner(pool).run(candidate, image, "Save")
    assert not event.get("error")
    assert event["prediction"]["point"] == [0.5, 0.5] and pool.calls == 2
    assert "icon 2" in event["stdout"]


def test_child_cannot_reset_parent_call_budget(tmp_path, monkeypatch):
    candidate, image = files(
        tmp_path,
        """def ground(screenshot,instruction,pool):\n for i in range(5):\n  pool.calls=0\n  try:pool.ask_vlm('qwen','Read')\n  except RuntimeError:break\n return {'point':[0.5,0.5]}\n""",
    )

    def post(url, *, json, timeout):
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"choices": [{"message": {"content": "OK"}}]}
        )

    monkeypatch.setattr(httpx, "post", post)
    pool = ModelPool(load_config(), max_calls=2)
    event = CandidateRunner(pool).run(candidate, image, "Save")
    assert not event.get("error") and pool.calls == 2


def test_candidate_timeout_kills_the_process_group(tmp_path):
    candidate, image = files(tmp_path, """def ground(screenshot,instruction,pool):\n while True:pass\n""")
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        CandidateRunner(ModelPool(load_config()), timeout=0.5).run(candidate, image, "Save")
    assert time.monotonic() - start < 5


def test_only_pixels_instruction_and_public_models_reach_child(tmp_path, monkeypatch):
    monkeypatch.setenv("EXAMPLE_SECRET", "never-pass-this")
    candidate, image = files(
        tmp_path,
        """import os\nfrom pathlib import Path\ndef ground(screenshot,instruction,pool):\n assert os.getenv('EXAMPLE_SECRET') is None\n assert os.getenv('CUDA_VISIBLE_DEVICES')==''\n assert Path(screenshot).name=='screenshot.png'\n assert {p.name for p in Path.cwd().iterdir()}=={'screenshot.png','submission.py','__pycache__'}\n return {'point':[0.5,0.5]}\n""",
    )
    event = CandidateRunner(ModelPool(load_config())).run(candidate, image, "Save")
    assert not event.get("error"), event.get("error")


def test_evaluation_keeps_failures_in_denominator_and_rejects_reused_output(tmp_path):
    candidate, image = files(
        tmp_path,
        """def ground(screenshot,instruction,pool):\n if instruction=='crash':raise RuntimeError('Broken candidate')\n return {'point':[0.5,0.5]}\n""",
    )
    manifest = tmp_path / "labels.jsonl"
    rows = [
        {"id": str(i), "image": image.name, "instruction": instruction, "bbox_xyxy": [190, 90, 210, 110]}
        for i, instruction in enumerate(["Save", "crash"])
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = tmp_path / "results"
    summary = evaluate(manifest, candidate, load_config(), output)
    assert summary["accuracy"] == 0.5 and summary["operational_failures"] == 1 and summary["examples"] == 2
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 2
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["candidate_sha256"]) == 64 and len(metadata["manifest_sha256"]) == 64
    with pytest.raises(FileExistsError):
        evaluate(manifest, candidate, load_config(), output)


def test_bad_manifest_fails_before_execution(tmp_path):
    candidate, image = files(
        tmp_path, """def ground(screenshot,instruction,pool):raise AssertionError('Must not run')"""
    )
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text(
        json.dumps({"id": "x", "image": image.name, "instruction": "Save", "bbox_xyxy": [0, 0, 800, 200]})
        + "\n"
    )
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="original-image pixels"):
        evaluate(manifest, candidate, load_config(), output)
    assert not output.exists()


def test_submission_can_define_standard_library_dataclasses(tmp_path):
    candidate, image = files(
        tmp_path,
        """from dataclasses import dataclass
@dataclass
class Point:
 x: float
 y: float
def ground(screenshot,instruction,pool):
 p=Point(0.5,0.5)
 return {'point':[p.x,p.y]}
""",
    )
    event = CandidateRunner(ModelPool(load_config())).run(candidate, image, "Save")
    assert not event.get("error"), event.get("error")
    assert event["prediction"]["point"] == [0.5, 0.5]


@pytest.mark.parametrize("point_source", ["[True,0.5]", '[float("nan"),0.5]', "[1.1,0.5]", "[0,0,1,1]"])
def test_invalid_submission_points_are_misses(tmp_path, point_source):
    candidate, image = files(
        tmp_path, f"def ground(screenshot,instruction,pool):\n return {{'point':{point_source}}}\n"
    )
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text(
        json.dumps({"image": image.name, "instruction": "Save", "bbox_xyxy": [0, 0, 400, 200]}) + "\n"
    )
    result = evaluate(manifest, candidate, load_config(), tmp_path / "output")
    assert result["accuracy"] == 0 and result["operational_failures"] == 1
