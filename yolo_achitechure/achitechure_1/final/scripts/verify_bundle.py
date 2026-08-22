#!/usr/bin/env python3
"""建立或驗證 final/MANIFEST.json。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _bundle import BUNDLE_ROOT, atomic_json, file_sha256

EXCLUDED_PARTS = {"__pycache__", "outputs"}
EXCLUDED_NAMES = {"MANIFEST.json", "queue-state.json"}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    destination = BUNDLE_ROOT / "MANIFEST.json"
    if args.write:
        atomic_json(destination, build_manifest())
        print(destination)
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
    print(f"PASS：{len(actual_paths)} 個檔案均符合 MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
