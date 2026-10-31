"""Bounded Modal experiments: one H100, three local servers, no public web endpoint.

Uses your authenticated Modal CLI configuration:
  modal run modal_app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).parent
app = modal.App("ensemble-desktop-localization")
cache = modal.Volume.from_name("ensemble-desktop-localization-hf-v1", create_if_missing=True)
results = modal.Volume.from_name("ensemble-desktop-localization-results-v1", create_if_missing=True)

base = (
    modal.Image.from_registry("vllm/vllm-openai:v0.32.0")
    .entrypoint([])
    .run_commands('ln -sf "$(command -v python3)" /usr/local/bin/python')
    .apt_install("fonts-dejavu-core")
    .uv_pip_install("Pillow>=11,<13", "httpx>=0.28,<0.29", "numpy>=2,<3")
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONPATH": "/root"})
    .add_local_dir(ROOT / "grounding", "/root/grounding")
    .add_local_file(ROOT / "models.json", "/root/models.json")
    .add_local_file(ROOT / "submission.py", "/root/submission.py")
    .add_local_file(ROOT / "task.toml", "/root/task.toml")
)


@app.function(image=base, cpu=8, memory=16384, timeout=1800, volumes={"/cache": cache})
def download_models():
    from huggingface_hub import snapshot_download

    from grounding.config import load_config

    config = load_config("/root/models.json")
    downloaded = []
    for model in config["models"]:
        print(f"Downloading pinned {model['alias']}: {model['repo_id']}@{model['revision']}", flush=True)
        path = snapshot_download(
            model["repo_id"],
            revision=model["revision"],
            ignore_patterns=["*.md", ".gitattributes", "*.pth", "*.pt", "original/*"],
        )
        downloaded.append({"alias": model["alias"], "snapshot": path})
        cache.commit()
    return downloaded


@app.function(
    image=base,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=3000,
    name="ensemble-desktop-localization",
    max_containers=1,
    volumes={"/cache": cache, "/results": results},
)
def test_h100(run_id: str, long_context: bool = True):
    import hashlib
    import importlib.metadata
    import platform
    import traceback

    import torch
    import vllm

    from grounding.config import load_config
    from grounding.evaluation import evaluate
    from grounding.screenspot import prepare_visible_sample
    from grounding.servers import MemoryMonitor, ServerGroup, gpu_memory
    from grounding.smoke import smoke

    cache.reload()
    config = load_config("/root/models.json")
    output = Path("/results") / run_id
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "hardware": gpu_memory(),
        "config": config,
        "status": "running",
    }
    report["submission_sha256"] = hashlib.sha256(Path("/root/submission.py").read_bytes()).hexdigest()
    report["packages"] = {
        name: importlib.metadata.version(name)
        for name in ["transformers", "huggingface_hub", "Pillow", "numpy", "httpx"]
    }
    monitor = MemoryMonitor(output / "gpu-memory.jsonl").start()
    try:
        with ServerGroup(config, output / "servers") as servers:
            report["startup_memory"] = servers.snapshots
            report["smoke"] = smoke(config, output, long_context=long_context)
            visible = prepare_visible_sample(output / "visible", count=3)
            report["visible_sample"] = evaluate(
                visible, "/root/submission.py", config, output / "visible-evaluation"
            )
            if report["visible_sample"]["operational_failures"]:
                raise RuntimeError("Visible integration sample had operational failures")
            report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        monitor.close()
        report["memory_monitor"] = monitor.summary()
        report["final_memory"] = gpu_memory()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        results.commit()
    return report


@app.local_entrypoint()
def main(download_only: bool = False, skip_download: bool = False, long_context: bool = True):
    if not skip_download:
        print(json.dumps(download_models.remote(), indent=2))
    if download_only:
        return
    run_id = datetime.now(timezone.utc).strftime("h100-%Y%m%dT%H%M%SZ")
    report = test_h100.remote(run_id, long_context)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    path = output / f"{run_id}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report saved to {path}; remote volume path /{run_id}/report.json")
    print(json.dumps({"run_id": run_id, "status": report["status"], "error": report.get("error")}, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("H100 smoke failed; inspect the saved report and server logs")
