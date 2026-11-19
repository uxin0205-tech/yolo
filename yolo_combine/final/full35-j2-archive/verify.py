#!/usr/bin/env python3
"""Verify every regular file in this final package against MANIFEST.json."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"MANIFEST.json", "CHECKSUMS.sha256"}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
expected = set()
total = 0
for record in payload["files"]:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe path: {relative}")
    path = ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"integrity mismatch: {relative}")
    expected.add(relative.as_posix())
    total += path.stat().st_size
actual = {
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file()
    and not path.is_symlink()
    and path.name not in EXCLUDED
    and "__pycache__" not in path.parts
    and path.suffix != ".pyc"
}
if actual != expected:
    raise ValueError(f"manifest coverage mismatch: missing={actual-expected}, extra={expected-actual}")
print(json.dumps({"valid": True, "files": len(expected), "bytes": total}, indent=2))
