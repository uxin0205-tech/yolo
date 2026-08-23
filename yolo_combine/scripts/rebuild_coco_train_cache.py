#!/usr/bin/env python3
"""Rebuild and validate the full COCO train2017 Ultralytics label cache.

This utility is deliberately model-free: it scans only the manifest, images, and
YOLO labels, writes a candidate cache, and records validation evidence. It never
loads a checkpoint or allocates a CUDA device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from ultralytics.data import dataset as dataset_module
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, img2label_paths, load_dataset_cache_file
from ultralytics.utils import YAML

EXPECTED_IMAGES = 118_287
EXPECTED_RESULTS = (117_266, 1_021, 0, 0, 118_287)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def manifest_images(manifest: Path) -> list[str]:
    base = manifest.resolve().parent
    images: list[str] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        path = base / value[2:] if value.startswith("./") else Path(value)
        if not path.is_absolute():
            path = base / path
        images.append(str(path.resolve()))
    return sorted(images)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must remain unavailable during cache recovery")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    images = manifest_images(args.manifest)
    if len(images) != EXPECTED_IMAGES or len(set(images)) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"manifest must contain {EXPECTED_IMAGES} unique images, got "
            f"{len(images)} entries and {len(set(images))} unique paths"
        )

    labels = img2label_paths(images)
    data = YAML.load(args.data)
    if len(data["names"]) != 80:
        raise RuntimeError(f"full COCO cache requires 80 classes, got {len(data['names'])}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    dataset_module.NUM_THREADS = args.workers

    dataset = YOLODataset.__new__(YOLODataset)
    dataset.im_files = images
    dataset.label_files = labels
    dataset.prefix = "COCO train2017 cache recovery: "
    dataset.use_keypoints = False
    dataset.data = data
    dataset.single_cls = False
    dataset.augment = False
    dataset.cache_labels(args.output)

    cache = load_dataset_cache_file(args.output)
    results = tuple(int(value) for value in cache["results"])
    cached_images = [entry["im_file"] for entry in cache["labels"]]
    expected_hash = get_hash(labels + images)
    checks = {
        "cuda_visible_devices_empty": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "cuda_unavailable": not torch.cuda.is_available(),
        "version_matches": cache.get("version") == DATASET_CACHE_VERSION,
        "results_match": results == EXPECTED_RESULTS,
        "label_entries_match": len(cache["labels"]) == EXPECTED_IMAGES,
        "manifest_order_matches": cached_images == images,
        "hash_matches": cache.get("hash") == expected_hash,
        "messages_empty": cache.get("msgs") == [],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"rebuilt cache failed validation: {failed}")

    report = {
        "manifest": str(args.manifest.resolve()),
        "data": str(args.data.resolve()),
        "cache": str(args.output.resolve()),
        "cache_bytes": args.output.stat().st_size,
        "cache_sha256": sha256(args.output),
        "cache_version": cache["version"],
        "dataset_hash": cache["hash"],
        "results": list(results),
        "label_entries": len(cache["labels"]),
        "workers": args.workers,
        "checks": checks,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
