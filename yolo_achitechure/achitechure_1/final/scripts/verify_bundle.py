#!/usr/bin/env python3
"""建立或驗證 final/MANIFEST.json。"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bundle import BUNDLE_ROOT, atomic_json, file_sha256, load_models

EXCLUDED_PARTS = {"__pycache__", "outputs"}
EXCLUDED_NAMES = {"MANIFEST.json", "queue-state.json"}
REQUIRED_TRAINING_FILES = {
    "manifest.json",
    "training-complete.json",
    "resource-telemetry.jsonl",
    "args.yaml",
    "results.csv",
    "results.png",
    "BoxF1_curve.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "labels.jpg",
}


def files() -> list[Path]:
    """列出固定交付內容，不含可再生 runtime output。"""

    return sorted(
        path
        for path in BUNDLE_ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.relative_to(BUNDLE_ROOT).parts)
        and path.suffix != ".pyc"
    )


def build_manifest() -> dict[str, object]:
    """計算目前 bundle manifest。"""

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle": "full35-partial75-final",
        "files": [
            {
                "path": path.relative_to(BUNDLE_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in files()
        ],
    }


def finite_numbers(value: Any) -> bool:
    """遞迴確認 JSON 內沒有 NaN／Inf。"""

    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    return True


def markdown_links(path: Path) -> list[Path]:
    """列出 Markdown 中應存在的相對檔案連結。"""

    targets = []
    for raw in re.findall(r"\[[^]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        target = raw.strip().split("#", 1)[0]
        if not target or "://" in target or target.startswith("#"):
            continue
        targets.append((path.parent / target).resolve())
    return targets


def audit_bundle() -> dict[str, int]:
    """交叉驗證候選、metrics、訓練證據、權重索引與文件連結。"""

    models = load_models()
    model_ids = [record["id"] for record in models]
    if len(model_ids) != len(set(model_ids)):
        raise RuntimeError("models.json 含重複 id")
    accuracy = json.loads((BUNDLE_ROOT / "reports/full35-partial75-ap.json").read_text(encoding="utf-8"))
    accuracy_models = accuracy["models"]
    if [record["id"] for record in accuracy_models] != model_ids:
        raise RuntimeError("accuracy report 與 models.json 候選順序不符")
    if not finite_numbers(accuracy):
        raise RuntimeError("accuracy report 含 NaN／Inf")
    for model in models:
        raw_root = BUNDLE_ROOT / "reports/raw" / model["id"]
        for name in ("coco-internal.json", "coco-canonical.json", "bbt5-internal.json"):
            if not (raw_root / name).is_file():
                raise RuntimeError(f"{model['id']} 缺少 {name}")

    training = json.loads((BUNDLE_ROOT / "reports/rtx5060ti-final-0824.json").read_text(encoding="utf-8"))
    if not finite_numbers(training):
        raise RuntimeError("training report 含 NaN／Inf")
    training_ids = [record["model_id"] for record in training["training_runs"]]
    expected_training_ids = [record["id"] for record in models if record["stage"] in ("b", "c")]
    if training_ids != expected_training_ids:
        raise RuntimeError("training summary 與 B/C 候選不符")
    for model_id in training_ids:
        root = BUNDLE_ROOT / "reports/training" / model_id
        missing = sorted(name for name in REQUIRED_TRAINING_FILES if not (root / name).is_file())
        if missing:
            raise RuntimeError(f"{model_id} 缺少訓練證據：{missing}")
    for relative in training["plots"]:
        plot = BUNDLE_ROOT / "reports" / relative
        if not plot.is_file() or plot.stat().st_size < 1024:
            raise RuntimeError(f"圖檔不存在或過小：{relative}")

    weights = json.loads((BUNDLE_ROOT / "weights/index.json").read_text(encoding="utf-8"))
    expected_weight_records = len(models) + sum(model["float"] is not None for model in models)
    if len(weights["records"]) != expected_weight_records:
        raise RuntimeError("權重索引筆數不符")
    for record in weights["records"]:
        checkpoint = BUNDLE_ROOT / record["uri"]
        if not checkpoint.is_file() or file_sha256(checkpoint) != record["sha256"]:
            raise RuntimeError(f"權重索引 SHA 不符：{record['id']}")

    inventory = json.loads((BUNDLE_ROOT / "checkpoint-inventory.json").read_text(encoding="utf-8"))
    if {record["tested_as"] for record in inventory["records"]} != set(model_ids) - {"a0"}:
        raise RuntimeError("checkpoint inventory 未涵蓋所有 Full35／Partial75 候選")
    queue = json.loads(
        (BUNDLE_ROOT / "reports/raw/queues/fraction10-phase-c-state.json").read_text(encoding="utf-8")
    )
    if queue["status"] != "completed" or queue["pending_phase_c"]:
        raise RuntimeError("Phase C queue 並非 completed")

    markdown_files = [BUNDLE_ROOT / "README.md", BUNDLE_ROOT / "VALIDATION.md"]
    markdown_files.extend((BUNDLE_ROOT / "reports").glob("*.md"))
    for markdown in markdown_files:
        for target in markdown_links(markdown):
            if not target.exists():
                raise RuntimeError(f"失效 Markdown link：{markdown.relative_to(BUNDLE_ROOT)} -> {target}")
    return {
        "models": len(models),
        "dataset_validations": len(models) * 2,
        "training_runs": len(training_ids),
        "weight_records": len(weights["records"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    audit = audit_bundle()
    destination = BUNDLE_ROOT / "MANIFEST.json"
    if args.write:
        atomic_json(destination, build_manifest())
        print(f"{destination}；audit={audit}")
        return 0
    expected = json.loads(destination.read_text(encoding="utf-8"))
    expected_files = {item["path"]: item for item in expected["files"]}
    actual_paths = {path.relative_to(BUNDLE_ROOT).as_posix(): path for path in files()}
    if set(expected_files) != set(actual_paths):
        raise RuntimeError("MANIFEST 檔案集合與目前 bundle 不一致")
    for relative, path in actual_paths.items():
        item = expected_files[relative]
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"MANIFEST 驗證失敗：{relative}")
    print(f"PASS：{len(actual_paths)} 個檔案均符合 MANIFEST.json；audit={audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
