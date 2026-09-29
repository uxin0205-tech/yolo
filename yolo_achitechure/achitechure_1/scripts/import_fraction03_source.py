#!/usr/bin/env python3
"""將舊電腦的 accepted Phase A2 Float checkpoint 匯入固定續跑位置。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("full35", "partial75"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--note", default="由舊工作站匯入的 accepted Phase A2 boundary")
    args = parser.parse_args()

    source = args.checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = PROJECT_ROOT / "inputs/continuation" / f"{args.variant}-accepted-a2"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "float-best.pt"
    source_sha256 = _sha256(source)
    if destination.is_file():
        if _sha256(destination) != source_sha256:
            raise FileExistsError(f"既有檔案內容不同，拒絕覆寫：{destination}")
    elif source != destination.resolve():
        temporary = destination.with_suffix(".pt.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "boundary": "accepted_after_phase_a2",
        "note": args.note,
        "float_checkpoint": {
            "path": destination.name,
            "sha256": source_sha256,
            "imported_from": str(source),
        },
    }
    descriptor = output_dir / "candidate.json"
    if descriptor.is_file():
        existing = json.loads(descriptor.read_text(encoding="utf-8"))
        existing_checkpoint = existing.get("float_checkpoint", {})
        if (
            existing.get("variant") != args.variant
            or existing.get("boundary") != "accepted_after_phase_a2"
            or existing_checkpoint.get("path") != destination.name
            or existing_checkpoint.get("sha256") != source_sha256
        ):
            raise FileExistsError(f"既有來源描述不同，拒絕覆寫：{descriptor}")
    else:
        descriptor.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"已匯入 {args.variant}: {descriptor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
