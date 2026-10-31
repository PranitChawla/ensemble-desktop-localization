from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


class PythonSession:
    def __init__(self, screenshot: str | Path, instruction: str, pool, *, timeout: float = 180):
        self.pool, self.timeout = pool, timeout
        self.directory = tempfile.TemporaryDirectory(prefix="grounding-python-")
        target = Path(self.directory.name) / "screenshot.png"
        from PIL import Image
        with Image.open(screenshot) as image:
            image.convert("RGB").save(target)
        # Deliberately exclude Modal, HF, GitHub, and other inherited credentials.
        environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                       "HOME": self.directory.name, "CUDA_VISIBLE_DEVICES": "",
                       "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(Path(__file__).with_name("python_worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=self.directory.name, env=environment, start_new_session=True,
        )
        self.events = queue.Queue()
        self.errors = []
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._send({"screenshot_path": str(target), "instruction": instruction, "models": list(pool.models),
                    "grounding_formats": {alias: model.get("grounding", {}) for alias, model in pool.models.items()}})
        event = self._next(time.monotonic() + 30)
        if event.get("type") != "ready":
            self.close()
            raise RuntimeError(f"Python tool failed to initialize: {event}")

    def _read_stdout(self):
        for line in self.process.stdout:
            try:
                self.events.put(json.loads(line))
            except ValueError:
                self.events.put({"type": "protocol_error", "line": line[:500]})
        self.events.put({"type": "closed"})

    def _read_stderr(self):
        for line in self.process.stderr:
            if len(self.errors) < 20:
                self.errors.append(line[-500:])

    def _send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def _next(self, deadline):
        try:
            return self.events.get(timeout=max(0.001, deadline - time.monotonic()))
        except queue.Empty:
            self.close()
            raise TimeoutError("Python tool exceeded its wall-time limit") from None

    def execute(self, code: str) -> dict:
        deadline = time.monotonic() + self.timeout
        self._send({"code": code})
        while True:
            event = self._next(deadline)
            if event["type"] in {"model_call", "ground_model"}:
                try:
                    if event["type"] == "ground_model":
                        result = self.pool.ground(event["model"], event["prompt"],
                                                  encoded_image=event["encoded_image"], frame=event.get("frame"))
                        reply = {"result": result}
                    else:
                        text = self.pool.ask(event["model"], event["prompt"],
                                             encoded_image=event.get("encoded_image"))
                        reply = {"text": text}
                except Exception as error:
                    reply = {"error": f"{type(error).__name__}: {error}"}
                self._send(reply)
            elif event["type"] == "result":
                return event
            else:
                raise RuntimeError(f"Python tool stopped: {event}; {''.join(self.errors)}")

    def close(self):
        if getattr(self, "process", None) and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)
        if getattr(self, "directory", None):
            self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

