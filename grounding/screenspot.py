"""Small, explicitly visible ScreenSpot-Pro plumbing sample; never loads OSWorld-G."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DATASET = "likaixin/ScreenSpot-Pro"
REVISION = "210e78d3844251110bff86c95835ebd37a6930fa"


def prepare_visible_sample(directory: str | Path, count: int = 3) -> Path:
    if not 1 <= count <= 55:
        raise ValueError("Initial visible sample supports 1–55 VS Code examples")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
    with urllib.request.urlopen(base + "/annotations/vscode_macos.json", timeout=60) as response:
        annotations = json.load(response)
    rows = []
    for row in annotations[:count]:
        relative = "images/" + row["img_filename"]
        destination = directory / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(base + "/" + relative, timeout=120) as response:
                destination.write_bytes(response.read())
        rows.append({"id": row["id"], "image": relative, "instruction": row["instruction"],
                     "bbox_xyxy": row["bbox"], "source": DATASET, "revision": REVISION})
    manifest = directory / "visible.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "provenance.json").write_text(json.dumps({
        "dataset": DATASET, "revision": REVISION, "selection": f"first {count} VS Code rows",
        "purpose": "visible integration sample, not representative benchmark evaluation",
    }, indent=2) + "\n")
    return manifest

