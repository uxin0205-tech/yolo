"""Re-evaluate canonical weights with COCO size-stratified AP metrics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .model import build_student
from .variants.definitions import variant_from_resolved_config


AREA_FIELDS = ("coco_mAP50_95", "coco_mAP50", "coco_mAP75", "mAPs", "mAPm", "mAPl")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_name(variant: str) -> str:
    return variant.replace("/", "-").replace("+", "-")


def _load_model(bundle: Path):
    resolved = _read_json(bundle / "resolved_config.json")
    variant = variant_from_resolved_config(resolved)
    model = build_student(bundle / "model.yaml", variant)
    payload = torch.load(bundle / "weight.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError(f"invalid canonical checkpoint: {bundle / 'weight.pt'}")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError(f"checkpoint manifest missing: {bundle / 'weight.pt'}")
    if manifest.get("variant_id") != variant.id or manifest.get("variant_config_hash") != variant.config_hash:
        raise RuntimeError(f"checkpoint identity mismatch: {variant.id}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _predict(model, *, data: Path, save_dir: Path, device: str, batch: int, workers: int) -> Path:
    from ultralytics.data.converter import coco80_to_coco91_class
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class CocoJsonValidator(DetectionValidator):
        """Write official COCO category IDs without invoking Ultralytics' implicit annotation path."""

        def init_metrics(self, runtime_model) -> None:
            super().init_metrics(runtime_model)
            self.class_map = coco80_to_coco91_class()
            # Keep eval_json disabled here. The explicit evaluator below uses
            # the audited annotation path instead of Ultralytics' inferred path.
            self.is_coco = False
            self.is_lvis = False

    save_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = save_dir / "predictions.json"
    if prediction_path.exists():
        prediction_path.unlink()
    validator = CocoJsonValidator(
        args={
            "data": str(data),
            "imgsz": 640,
            "batch": batch,
            "workers": workers,
            "device": device,
            "split": "val",
            "plots": False,
            "rect": True,
            "save_json": True,
            "verbose": False,
        },
        save_dir=save_dir,
    )
    validator(model=model)
    if not prediction_path.is_file():
        raise RuntimeError(f"validator did not produce predictions: {prediction_path}")
    return prediction_path


def _coco_evaluate(annotation: Path, predictions: Path) -> dict[str, float]:
    from faster_coco_eval import COCO, COCOeval_faster

    ground_truth = COCO(annotation)
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval_faster(ground_truth, detections, iouType="bbox", print_function=lambda _line: None)
    evaluator.params.imgIds = sorted(ground_truth.getImgIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = evaluator.stats_as_dict
    return {
        "coco_mAP50_95": values["AP_all"],
        "coco_mAP50": values["AP_50"],
        "coco_mAP75": values["AP_75"],
        "mAPs": values["AP_small"],
        "mAPm": values["AP_medium"],
        "mAPl": values["AP_large"],
    }


def _apply_metrics(root: Path, archive_row: dict[str, Any], record: dict[str, Any]) -> None:
    updates = {key: record[key] for key in AREA_FIELDS}
    updates.update(
        {
            "area_metrics_evaluator": record["evaluator"],
            "area_metrics_annotation": record["annotation"],
            "area_metrics_annotation_sha256": record["annotation_sha256"],
            "area_metrics_checkpoint_sha256": record["checkpoint_sha256"],
        }
    )
    run_metrics_path = root / str(archive_row["source_run"]) / "validation_metrics.json"
    run_metrics = _read_json(run_metrics_path)
    run_metrics.update(updates)
    _write_json(run_metrics_path, run_metrics)

    bundle_metrics_path = root / Path(str(archive_row["archived_weight"])).parent / "validation_metrics.json"
    bundle_metrics = _read_json(bundle_metrics_path)
    bundle_metrics.update(updates)
    _write_json(bundle_metrics_path, bundle_metrics)


def _write_summary(output: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "format": "YOLO11 BinaryAttention COCO size-stratified AP evaluation",
        "variant_count": len(records),
        "area_ranges": {
            "small": "area < 32^2 pixels",
            "medium": "32^2 <= area < 96^2 pixels",
            "large": "area >= 96^2 pixels",
        },
        "records": records,
    }
    _write_json(output / "coco_area_metrics.json", payload)
    fields = list(records[0]) if records else ["variant"]
    with (output / "coco_area_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def evaluate_canonical_area_metrics(
    root: Path,
    *,
    annotation: Path,
    data: Path,
    device: str = "0",
    batch: int = 16,
    workers: int = 8,
    selected_variants: set[str] | None = None,
    keep_predictions: bool = False,
    force: bool = False,
) -> Path:
    root = root.resolve()
    annotation = annotation.resolve()
    data = data.resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not data.is_file():
        raise FileNotFoundError(data)
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run area evaluation with GPU access")

    archive = _read_json(root / "artifacts" / "final_weights" / "manifest.json")
    rows = archive.get("weights")
    if archive.get("variant_count") != 26 or not isinstance(rows, list) or len(rows) != 26:
        raise RuntimeError("canonical 26-weight archive is incomplete")

    annotation_hash = _sha256(annotation)
    work_root = root / "artifacts" / "area_metrics"
    report_root = root / "artifacts" / "reports"
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        variant = str(row["variant"])
        if selected_variants and variant not in selected_variants:
            continue
        bundle = root / Path(str(row["archived_weight"])).parent
        checkpoint = bundle / "weight.pt"
        checkpoint_hash = _sha256(checkpoint)
        if checkpoint_hash != row.get("sha256"):
            raise RuntimeError(f"canonical checkpoint hash mismatch: {variant}")
        variant_dir = work_root / _artifact_name(variant)
        result_path = variant_dir / "coco_area_metrics.json"
        record = _read_json(result_path) if result_path.exists() else {}
        reusable = (
            not force
            and all(isinstance(record.get(key), (int, float)) for key in AREA_FIELDS)
            and record.get("annotation_sha256") == annotation_hash
            and record.get("checkpoint_sha256") == checkpoint_hash
        )
        if reusable:
            print(f"[{index:02d}/26] {variant}: reuse")
        else:
            print(f"[{index:02d}/26] {variant}: validating", flush=True)
            model = _load_model(bundle)
            predictions = _predict(
                model,
                data=data,
                save_dir=variant_dir,
                device=device,
                batch=batch,
                workers=workers,
            )
            metrics = _coco_evaluate(annotation, predictions)
            native_metrics = _read_json(bundle / "validation_metrics.json")
            record = {
                "variant": variant,
                **metrics,
                "saved_mAP50_95": native_metrics.get("mAP50_95"),
                "coco_vs_saved_mAP_delta": (
                    metrics["coco_mAP50_95"] - float(native_metrics["mAP50_95"])
                    if isinstance(native_metrics.get("mAP50_95"), (int, float))
                    else None
                ),
                "seed": 0,
                "imgsz": 640,
                "split": "COCO2017 val2017",
                "image_count": 5000,
                "evaluator": "faster-coco-eval 1.7.2 / COCO bbox",
                "annotation": str(annotation.relative_to(root.parent)),
                "annotation_sha256": annotation_hash,
                "checkpoint": str(checkpoint.relative_to(root)),
                "checkpoint_sha256": checkpoint_hash,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(result_path, record)
            if not keep_predictions:
                predictions.unlink()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _apply_metrics(root, row, record)
        records.append(record)
        _write_summary(report_root, records)
    return report_root / "coco_area_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--annotation", type=Path, default=Path("../coco2017/annotations/instances_val2017.json"))
    parser.add_argument("--data", type=Path, default=Path("data/coco-full.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--variants", default=None, help="comma-separated canonical variant IDs")
    parser.add_argument("--keep-predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = set(args.variants.split(",")) if args.variants else None
    print(
        evaluate_canonical_area_metrics(
            args.root,
            annotation=args.annotation,
            data=args.data,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            selected_variants=selected,
            keep_predictions=args.keep_predictions,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
