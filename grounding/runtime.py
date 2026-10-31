"""Per-example subprocess and authoritative parent-owned VLM request accounting."""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image


class CandidateRunner:
    def __init__(self, pool, *, timeout=120):
        self.pool, self.timeout = pool, timeout

    def run(self, candidate, screenshot, instruction):
        with tempfile.TemporaryDirectory(prefix="gui-submission-") as directory:
            directory = Path(directory)
            source = directory / "submission.py"
            shutil.copyfile(candidate, source)
            target = directory / "screenshot.png"
            with Image.open(screenshot) as image:
                image.convert("RGB").save(target)
            models = {
                alias: {key: value for key, value in model.items() if key not in ["port"]}
                for alias, model in self.pool.models.items()
            }
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(directory),
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
            }
            process = subprocess.Popen(
                [sys.executable, "-I", "-u", str(Path(__file__).with_name("candidate_worker.py"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=directory,
                env=env,
                start_new_session=True,
            )
            events = queue.Queue()
            stderr = []

            def read_output():
                for line in process.stdout:
                    try:
                        events.put(json.loads(line))
                    except ValueError:
                        events.put({"type": "protocol_error", "text": line[:500]})
                events.put({"type": "closed"})

            def read_errors():
                for line in process.stderr:
                    if len(stderr) < 20:
                        stderr.append(line[-500:])

            readers = [
                threading.Thread(target=read_output, daemon=True),
                threading.Thread(target=read_errors, daemon=True),
            ]
            for reader in readers:
                reader.start()

            def send(value):
                process.stdin.write(json.dumps(value, allow_nan=False) + "\n")
                process.stdin.flush()

            deadline = time.monotonic() + self.timeout
            original_timeout = self.pool.timeout
            try:
                send(
                    {
                        "candidate": str(source),
                        "screenshot": str(target),
                        "instruction": instruction,
                        "models": models,
                        "max_calls": self.pool.max_calls,
                    }
                )
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Submission exceeded its per-example wall-time budget")
                    try:
                        event = events.get(timeout=remaining)
                    except queue.Empty:
                        raise TimeoutError("Submission exceeded its per-example wall-time budget") from None
                    if event["type"] == "model_call":
                        self.pool.timeout = min(original_timeout, max(0.01, deadline - time.monotonic()))
                        try:
                            if event["method"] not in ["chat", "ground"]:
                                raise ValueError("Unsupported VLM operation")
                            value = getattr(self.pool, event["method"])(event["alias"], **event["kwargs"])
                            send({"result": value, "calls": self.pool.calls})
                        except Exception as error:
                            send({"error": f"{type(error).__name__}: {error}", "calls": self.pool.calls})
                    elif event["type"] == "result":
                        return event
                    else:
                        raise RuntimeError(f"Submission process failed: {event}; {''.join(stderr)}")
            finally:
                self.pool.timeout = original_timeout
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                for reader in readers:
                    reader.join(timeout=2)
                for stream in [process.stdin, process.stdout, process.stderr]:
                    stream.close()
