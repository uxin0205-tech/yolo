#!/usr/bin/env python3
"""安全重建並校驗 GitHub 分片權重。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reconstruct(weight_id: str, output: Path | None, force: bool) -> Path:
    manifest: dict[str, Any] = json.loads(
        (ROOT / "weights.json").read_text(encoding="utf-8")
    )
    try:
        entry = manifest["weights"][weight_id]
    except KeyError as error:
        choices = ", ".join(sorted(manifest["weights"]))
        raise SystemExit(f"未知 weight id {weight_id!r}；可用值：{choices}") from error

    target = output or (ROOT / weight_id / entry["output_filename"])
    if target.exists() and not force:
        raise SystemExit(f"拒絕覆蓋既有檔案：{target}；確認後可加 --force")
    target.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    total_bytes = 0
    with target.open("wb") as destination:
        for part in entry["parts"]:
            source = ROOT / part["path"]
            if source.stat().st_size != part["bytes"]:
                raise SystemExit(f"分片大小不符：{source}")
            if _sha256(source) != part["sha256"]:
                raise SystemExit(f"分片 SHA-256 不符：{source}")
            with source.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    destination.write(block)
                    digest.update(block)
                    total_bytes += len(block)

    if total_bytes != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        target.unlink(missing_ok=True)
        raise SystemExit("重建後的大小或 SHA-256 不符；已移除無效輸出")
    print(
        json.dumps(
            {"output": str(target), "bytes": total_bytes, "sha256": digest.hexdigest()}
        )
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weight_id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    reconstruct(args.weight_id, args.output, args.force)


if __name__ == "__main__":
    main()
