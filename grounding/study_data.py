"""Reproducible, screenshot-disjoint ScreenSpot-Pro development/validation sample."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grounding.screenspot import DATASET, REVISION


def stable_key(text, seed=20261031):
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def balanced_sample(rows, count):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["application"], row.get("ui_type", "unknown"))].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: stable_key(row["id"]))
    selected = []
    keys = sorted(buckets, key=lambda key: stable_key(str(key)))
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def make_splits(rows, dev_count=128, validation_count=512):
    by_app = defaultdict(set)
    for row in rows:
        by_app[row["application"]].add(row["img_filename"])
    # Exclude the previous integration screenshots from confirmatory evaluation.
    prior_debug = {r["img_filename"] for r in rows if r["id"] in {"vscode_macos_0", "vscode_macos_1", "vscode_macos_2"}}
    dev_images = set(prior_debug)
    for app, images in by_app.items():
        ordered = sorted(images, key=lambda image: stable_key(app + image))
        dev_images.update(ordered[:max(1, int(len(ordered) * 0.35))])
    dev = balanced_sample([r for r in rows if r["img_filename"] in dev_images], dev_count)
    validation = balanced_sample([r for r in rows if r["img_filename"] not in dev_images], validation_count)
    if len(dev) < dev_count or len(validation) < validation_count:
        raise ValueError(f"Insufficient eligible examples: dev={len(dev)}, validation={len(validation)}")
    assert not ({r['img_filename'] for r in dev} & {r['img_filename'] for r in validation})
    return {"dev": dev, "validation": validation}


def prepare_study(directory, dev_count=128, validation_count=512):
    from huggingface_hub import HfApi, hf_hub_download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    annotation_files = [name for name in files if name.startswith("annotations/") and name.endswith(".json")]

    def download(name):
        return Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION))

    rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for path in executor.map(download, annotation_files):
            rows.extend(json.loads(path.read_text()))
    splits = make_splits(rows, dev_count, validation_count)
    image_names = sorted({"images/" + row["img_filename"] for split in splits.values() for row in split})
    print(f"Preparing {len(rows)} annotations, {len(image_names)} selected screenshots", flush=True)
    with ThreadPoolExecutor(max_workers=12) as executor:
        for name, path in zip(image_names, executor.map(download, image_names)):
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.symlink_to(path)
    metadata = {"dataset": DATASET, "revision": REVISION, "seed": 20261031,
                "source_examples": len(rows), "screenshots_downloaded": len(image_names),
                "selection": "Application/UI-type round robin; screenshot-disjoint development/validation",
                "prior_debug_images_excluded_from_validation": sorted(prior_debug_images(rows)), "splits": {}}
    for name, split in splits.items():
        manifest = []
        for row in split:
            item = dict(row)
            item.update(image="images/" + row["img_filename"], bbox_xyxy=row["bbox"],
                        source=DATASET, revision=REVISION)
            manifest.append(item)
        text = "".join(json.dumps(row) + "\n" for row in manifest)
        (directory / f"{name}.jsonl").write_text(text)
        metadata["splits"][name] = {"examples": len(split),
            "screenshots": len({r["img_filename"] for r in split}),
            "sha256": hashlib.sha256(text.encode()).hexdigest()}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prior_debug_images(rows):
    return {r["img_filename"] for r in rows if r["id"] in {"vscode_macos_0", "vscode_macos_1", "vscode_macos_2"}}

