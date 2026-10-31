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
        rows.append(
            {
                "id": row["id"],
                "image": relative,
                "instruction": row["instruction"],
                "bbox_xyxy": row["bbox"],
                "source": DATASET,
                "revision": REVISION,
            }
        )
    manifest = directory / "visible.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "selection": f"first {count} VS Code rows",
                "purpose": "visible integration sample, not representative benchmark evaluation",
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def prepare_visible(directory: str | Path, limit: int | None = None) -> Path:
    """Prepare the pinned visible set on CPU, with no model requests."""
    import hashlib
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = HfApi().list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    annotations = []
    for filename in sorted(
        name for name in files if name.startswith("annotations/") and name.endswith(".json")
    ):
        source = hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION)
        annotations.extend(json.loads(Path(source).read_text()))
    if limit is not None:
        annotations = annotations[:limit]
    if not annotations:
        raise ValueError("No visible annotations found at the pinned revision")
    image_names = sorted({"images/" + row["img_filename"] for row in annotations})

    def download(filename):
        source = hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION)
        destination = directory / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(download, image_names))
    rows = [
        dict(
            id=row["id"],
            image="images/" + row["img_filename"],
            instruction=row["instruction"],
            bbox_xyxy=row["bbox"],
            source=DATASET,
            revision=REVISION,
        )
        for row in annotations
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    manifest = directory / "visible.jsonl"
    manifest.write_text(text)
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "examples": len(rows),
                "selection": "all sorted annotation files, optional fixed prefix",
                "manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return manifest
