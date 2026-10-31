from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx


def gpu_memory() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.strip().splitlines():
        name, total, used, free = [v.strip() for v in line.split(",")]
        rows.append({"name": name, "total_mib": int(total), "used_mib": int(used), "free_mib": int(free)})
    return {"gpus": rows}


class MemoryMonitor:
    """Sample device memory through startup, prefill, decoding, and Python calls."""

    def __init__(self, path: str | Path, interval=1.0):
        self.path, self.interval = Path(path), interval
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.path.write_text("")
        while not self.stop_event.is_set():
            try:
                sample = {"time": time.time(), **gpu_memory()}
                self.samples.append(sample)
                with self.path.open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                with self.path.open("a") as stream:
                    stream.write(json.dumps({"error": str(error)}) + "\n")
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def summary(self):
        devices = [sample["gpus"][0] for sample in self.samples if sample.get("gpus")]
        return {
            "samples": len(devices),
            "interval_seconds": self.interval,
            "sampled_peak_used_mib": max((d["used_mib"] for d in devices), default=None),
            "sampled_min_free_mib": min((d["free_mib"] for d in devices), default=None),
        }


def server_command(config: dict, model: dict) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model["repo_id"],
        "--revision",
        model["revision"],
        "--served-model-name",
        model["alias"],
        "--host",
        "127.0.0.1",
        "--port",
        str(model["port"]),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(model["gpu_memory_utilization"]),
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(model.get("max_num_batched_tokens", config["max_num_batched_tokens"])),
        "--limit-mm-per-prompt",
        json.dumps({"image": 3, "video": 0}),
        "--enforce-eager",
        "--seed",
        "0",
    ]
    if model.get("server_processor_kwargs", model.get("processor_kwargs")):
        command.extend(
            [
                "--mm-processor-kwargs",
                json.dumps(model.get("server_processor_kwargs", model["processor_kwargs"])),
            ]
        )
    command.extend(model.get("extra_server_args", []))
    return command


class ServerGroup:
    def __init__(self, config: dict, log_dir: str | Path, *, startup_timeout=900):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = startup_timeout
        self.processes = []
        self.snapshots = []

    def start(self):
        self.snapshots.append({"stage": "before_servers", **gpu_memory()})
        for model in self.config["models"]:
            log_path = self.log_dir / f"{model['alias']}.log"
            log = log_path.open("w")
            command = server_command(self.config, model)
            environment = dict(os.environ)
            environment.update(
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="2",
                HF_HUB_OFFLINE="1",
            )
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
            )
            self.processes.append((process, log))
            print(f"Starting {model['alias']} on localhost:{model['port']}", flush=True)
            deadline = time.monotonic() + self.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{model['alias']} exited {process.returncode}:\n{log_path.read_text()[-10000:]}"
                    )
                try:
                    response = httpx.get(f"http://127.0.0.1:{model['port']}/health", timeout=2)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{model['alias']} startup timeout:\n{log_path.read_text()[-10000:]}")
                time.sleep(2)
            snapshot = {"stage": f"after_{model['alias']}", **gpu_memory()}
            self.snapshots.append(snapshot)
            print(json.dumps(snapshot), flush=True)
        return self

    def close(self):
        for process, _ in reversed(self.processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in reversed(self.processes):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            log.close()

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()
