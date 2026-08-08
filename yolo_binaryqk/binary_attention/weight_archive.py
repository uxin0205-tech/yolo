"""Export one durable, audited weight bundle for every canonical ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch

from .model import build_student
from .variants.definitions import variant_from_resolved_config
from .verification.checkpoint import strict_reload


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


def _artifact_name(variant: str) -> str:
    return variant.replace("/", "-").replace("+", "-")


def _preserve_weight(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != _sha256(source):
            raise RuntimeError(f"existing archived weight disagrees with source: {destination}")
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    source_stat = source.stat()
    destination_stat = destination.stat()
    return "hardlink" if (source_stat.st_dev, source_stat.st_ino) == (destination_stat.st_dev, destination_stat.st_ino) else "copy"


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["variant"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _verify_archived_weight(weight: Path, model_yaml: Path, resolved: dict) -> str:
    definition = variant_from_resolved_config(resolved)
    model = build_student(model_yaml, definition)
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    if not isinstance(manifest, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError(f"invalid strict checkpoint payload: {weight}")
    if manifest.get("variant_id") != definition.id or manifest.get("variant_config_hash") != definition.config_hash:
        raise RuntimeError(f"checkpoint variant/config hash mismatch: {weight}")
    if int(manifest.get("experiment_schema_version", 0)) >= 3:
        strict_reload(weight, model, definition)
        return "schema3_strict_reload"
    # E-series artifacts predate the schema-3 quantization-contract field.
    # Preserve their immutable schema-2 manifest while still requiring a
    # strict state-dict load and exact variant/config identity.
    model.load_state_dict(payload["state_dict"], strict=True)
    return "legacy_schema2_strict_state_dict_reload"


def export_canonical_weights(root: Path, *, verify_load: bool = False) -> Path:
    root = root.resolve()
    audit_path = root / "logs" / "formal-plan" / "final-audit.json"
    audit = _read_json(audit_path)
    if audit.get("ok") is not True or audit.get("verified_variant_count") != audit.get("expected_variant_count"):
        raise RuntimeError("final audit must pass before canonical weights can be exported")
    runs = audit.get("runs")
    if not isinstance(runs, dict) or len(runs) != 26:
        raise RuntimeError("final audit does not identify exactly 26 canonical runs")

    output = root / "artifacts" / "final_weights"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, relative_run in runs.items():
        run = root / str(relative_run)
        status = _read_json(run / "status.json")
        resolved = _read_json(run / "resolved_config.json")
        metrics = _read_json(run / "validation_metrics.json")
        source_name = "strict-validation.pt" if status.get("stage") == "validation" else "strict-last.pt"
        source_weight = run / "checkpoints" / source_name
        if not source_weight.is_file():
            raise FileNotFoundError(source_weight)

        variant_dir = output / _artifact_name(str(variant))
        archived_weight = variant_dir / "weight.pt"
        storage_mode = _preserve_weight(source_weight, archived_weight)
        for name in (
            "model.yaml", "resolved_config.json", "validation_metrics.json",
            "checkpoint_manifest.json", "status.json",
        ):
            shutil.copy2(run / name, variant_dir / name)

        checkpoint_hash = _sha256(source_weight)
        archived_hash = _sha256(archived_weight)
        if checkpoint_hash != archived_hash:
            raise RuntimeError(f"archived weight hash mismatch: {variant}")
        verification_mode = _verify_archived_weight(
            archived_weight, variant_dir / "model.yaml", resolved
        ) if verify_load else "not_run"

        rows.append({
            "variant": variant,
            "stage": status.get("stage"),
            "mAP50_95": metrics.get("mAP50_95"),
            "coco_mAP50_95": metrics.get("coco_mAP50_95"),
            "mAPs": metrics.get("mAPs"),
            "mAPm": metrics.get("mAPm"),
            "mAPl": metrics.get("mAPl"),
            "archived_weight": str(archived_weight.relative_to(root)),
            "source_run": str(run.relative_to(root)),
            "source_weight": str(source_weight.relative_to(root)),
            "model_yaml": str((variant_dir / "model.yaml").relative_to(root)),
            "config_hash": resolved.get("config_hash"),
            "sha256": archived_hash,
            "size_bytes": archived_weight.stat().st_size,
            "storage_mode": storage_mode,
            "strict_reload_verified": bool(verify_load),
            "verification_mode": verification_mode,
            "checkpoint_weight_source": status.get("checkpoint_weight_source"),
            "metrics_weight_source": status.get("metrics_weight_source"),
            "epochs": status.get("epochs"),
            "trainable_scope": status.get("trainable_scope"),
        })

    manifest = {
        "format": "YOLO11 BinaryAttention canonical ablation weight archive",
        "variant_count": len(rows),
        "source_audit": str(audit_path.relative_to(root)),
        "source_audit_ok": True,
        "weights": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _write_csv(output / "manifest.csv", rows)
    lines = [
        "# Canonical BinaryAttention ablation weights",
        "",
        "每個目錄保留一個與正式 validation metrics 對應的 strict weight，以及重建模型所需的 YAML/JSON metadata。",
        "Ultralytics optimizer/resume checkpoints 可刪除；本目錄的 `weight.pt` 不可刪除。",
        "",
        "| Variant | 原 mAP50–95 | COCO mAP | APs | APm | APl | Weight | SHA-256 | Strict reload |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['mAP50_95']} | {row['coco_mAP50_95']} | "
            f"{row['mAPs']} | {row['mAPm']} | {row['mAPl']} | `{row['archived_weight']}` | "
            f"`{row['sha256']}` | {row['strict_reload_verified']} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-load", action="store_true")
    args = parser.parse_args()
    print(export_canonical_weights(args.root, verify_load=args.verify_load))


if __name__ == "__main__":
    main()
