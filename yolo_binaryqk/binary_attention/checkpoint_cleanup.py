"""Delete redundant Ultralytics resume checkpoints after canonical export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .formal_audit import audit_formal_plan


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive(root: Path) -> dict:
    audit_path = root / "logs" / "formal-plan" / "final-audit.json"
    audit = audit_formal_plan(root)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    archive = _read_json(root / "artifacts" / "final_weights" / "manifest.json")
    rows = archive.get("weights")
    if audit.get("ok") is not True or audit.get("archived_weight_count") != 26:
        raise RuntimeError("final audit with 26 archived weights must pass before cleanup")
    if archive.get("variant_count") != 26 or not isinstance(rows, list) or len(rows) != 26:
        raise RuntimeError("canonical weight archive is incomplete")
    for row in rows:
        if row.get("strict_reload_verified") is not True:
            raise RuntimeError(f"archived weight was not reload-verified: {row.get('variant')}")
        weight = root / str(row.get("archived_weight", ""))
        if not weight.is_file() or weight.stat().st_size != row.get("size_bytes"):
            raise RuntimeError(f"archived weight missing or wrong size: {row.get('variant')}")
        if _sha256(weight) != row.get("sha256"):
            raise RuntimeError(f"archived weight hash mismatch: {row.get('variant')}")
    return archive


def cleanup_training_checkpoints(root: Path, *, execute: bool = False) -> dict:
    root = root.resolve()
    _verify_archive(root)
    runs_root = (root / "artifacts" / "runs").resolve()
    targets = sorted(
        path.resolve()
        for path in runs_root.glob("**/ultralytics/train/weights/*.pt")
        if path.name in {"best.pt", "last.pt"}
    )
    if any(not target.is_relative_to(runs_root) for target in targets):
        raise RuntimeError("cleanup target escaped artifacts/runs")
    entries = [
        {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size}
        for path in targets
    ]
    result = {
        "status": "planned",
        "recoverability": (
            "optimizer/resume state is not recoverable after deletion except by retraining; "
            "all 26 canonical strict weights and reconstruction metadata remain preserved"
        ),
        "target_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "targets": entries,
    }
    manifest = root / "artifacts" / "final_weights" / "checkpoint_cleanup.json"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if execute:
        for target in targets:
            target.unlink()
        result.update({
            "status": "complete",
            "deleted_count": len(entries),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(cleanup_training_checkpoints(args.root, execute=args.execute), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
