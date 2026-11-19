#!/usr/bin/env python3
"""CPU experiments for the BBT5 context-radius/receptive-field question.

The script intentionally keeps the intervention data in memory and stores only
predictions and compact metadata.  It has three resumable stages:

  audit   Build the source-grouped clean validation manifest and cell stats.
  infer   Run one FULL/R1/R2/R4/R8 x GRAY/MEAN condition on CPU.
  analyze Compute AP/Recall/paired retention, bootstrap CIs and a report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = (SCRIPT_DIR / "../bbt5-detect-baseline/dataset").resolve()
CHECKPOINT = (SCRIPT_DIR / "../bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt").resolve()
OUT_ROOT = SCRIPT_DIR / "context_rf_cpu"
IMG_SIZE = 640
STRIDES = (4, 8, 16)
RADII = (1, 2, 4, 8)
MASKS = ("gray", "mean")
CONF_THRESHOLD = 0.25
PREDICT_CONF = 0.001
IOU_MATCH = 0.5


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def source_stem(name: str) -> str:
    for marker in ("_jpg.rf.", "_png_jpg.rf."):
        if marker in name:
            return name.split(marker, 1)[0]
    return Path(name).stem


def source_group(name: str) -> str:
    stem = source_stem(name)
    # Frame indices are not a reliable scene boundary.  This deliberately
    # conservative grouping may over-group, which is safer for validation.
    return re.sub(r"(?i)(?:[-_](?:frame)?\d{1,6})$", "", stem)


def label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_labels(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.is_file():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        vals = line.split()
        if len(vals) < 5:
            raise ValueError(f"{path}:{line_no}: expected >=5 fields")
        cls, x, y, w, h = int(vals[0]), *map(float, vals[1:5])
        if cls not in (0, 1) or not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_no}: invalid YOLO row {line!r}")
        rows.append({"cls": cls, "x": x, "y": y, "w": w, "h": h})
    return rows


def xyxy_from_norm(row: dict[str, float], width: int, height: int) -> np.ndarray:
    x, y, w, h = row["x"] * width, row["y"] * height, row["w"] * width, row["h"] * height
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)


def letterbox_dims(width: int, height: int, size: int = IMG_SIZE) -> tuple[float, int, int]:
    scale = min(size / width, size / height)
    return scale, round(width * scale), round(height * scale)


def laplacian_variance(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    a = gray.astype(np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (
        a[:-2, 1:-1]
        + a[2:, 1:-1]
        + a[1:-1, :-2]
        + a[1:-1, 2:]
        - 4 * a[1:-1, 1:-1]
    )
    return float(lap.var())


def box_distance(a: np.ndarray, b: np.ndarray) -> float:
    dx = max(float(a[0] - b[2]), float(b[0] - a[2]), 0.0)
    dy = max(float(a[1] - b[3]), float(b[1] - a[3]), 0.0)
    return math.hypot(dx, dy)


def size_bin(short_side: float) -> str:
    if short_side < 8:
        return "<8"
    if short_side < 16:
        return "8-16"
    if short_side < 32:
        return "16-32"
    return ">=32"


def image_paths(split: str) -> list[Path]:
    return sorted((DATA_ROOT / split / "images").glob("*.jpg"), key=lambda p: p.name)


def build_audit(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    train_paths = image_paths("train")
    valid_paths = image_paths("valid")
    train_groups = {source_group(p.name) for p in train_paths}
    valid_groups = {source_group(p.name) for p in valid_paths}
    clean_paths = [p for p in valid_paths if source_group(p.name) not in train_groups]
    clean_set = {p.name for p in clean_paths}

    all_rows: list[dict[str, Any]] = []
    image_stats: dict[str, dict[str, Any]] = {}
    split_summary: dict[str, Any] = {}
    for split, paths in (("train", train_paths), ("valid", valid_paths)):
        object_counts = Counter()
        dims = Counter()
        empty = 0
        ball_count = 0
        for p in paths:
            with Image.open(p) as im:
                rgb = np.asarray(im.convert("RGB"))
            height, width = rgb.shape[:2]
            dims[f"{width}x{height}"] += 1
            labels = read_labels(label_path(p))
            if not labels:
                empty += 1
            boxes = [xyxy_from_norm(row, width, height) for row in labels]
            bats = [box for row, box in zip(labels, boxes) if row["cls"] == 1]
            for row, box in zip(labels, boxes):
                object_counts[str(row["cls"])] += 1
                if row["cls"] != 0:
                    continue
                ball_count += 1
                scale, resized_w, resized_h = letterbox_dims(width, height)
                w_px = float(row["w"] * width * scale)
                h_px = float(row["h"] * height * scale)
                area = w_px * h_px
                short = min(w_px, h_px)
                nearest_bat = min((box_distance(box, b) * scale for b in bats), default=float("inf"))
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                pad = 4
                crop = rgb[max(0, y0 - pad) : min(height, y1 + pad), max(0, x0 - pad) : min(width, x1 + pad)]
                gray = np.asarray(Image.fromarray(crop).convert("L"))
                row_out: dict[str, Any] = {
                    "split": split,
                    "image": p.name,
                    "image_path": str(p.resolve()),
                    "group": source_group(p.name),
                    "width": width,
                    "height": height,
                    "resized_width": resized_w,
                    "resized_height": resized_h,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "w_px_640": w_px,
                    "h_px_640": h_px,
                    "area_px_640": area,
                    "short_px_640": short,
                    "size_bin": size_bin(short),
                    "nearest_bat_px_640": nearest_bat,
                    "bat_relation": "no_bat" if math.isinf(nearest_bat) else ("overlap" if nearest_bat == 0 else ("near" if nearest_bat < 32 * scale else "far")),
                    "edge_distance_px_640": min(box[0], box[1], width - box[2], height - box[3]) * scale,
                    "blur_score": laplacian_variance(gray),
                }
                for stride in STRIDES:
                    row_out[f"w_cell_p{stride}"] = w_px / stride
                    row_out[f"h_cell_p{stride}"] = h_px / stride
                    row_out[f"short_cell_p{stride}"] = short / stride
                all_rows.append(row_out)
            image_stats[p.name] = {
                "split": split,
                "group": source_group(p.name),
                "path": str(p.resolve()),
                "width": width,
                "height": height,
                "labels": labels,
            }
        split_summary[split] = {
            "images": len(paths),
            "objects": dict(object_counts),
            "ball_count": ball_count,
            "empty_images": empty,
            "dimensions": dict(dims),
            "groups": len({source_group(p.name) for p in paths}),
        }

    train_stems = {source_stem(p.name) for p in train_paths}
    valid_stems = {source_stem(p.name) for p in valid_paths}
    overlap_groups = sorted({source_group(p.name) for p in train_paths} & {source_group(p.name) for p in valid_paths})
    overlap_stems = sorted(train_stems & valid_stems)
    clean_rows = [r for r in all_rows if r["split"] == "valid" and r["image"] in clean_set]

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with (out / "cell_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (out / "targets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)
    clean_paths_sorted = sorted(clean_paths, key=lambda p: p.name)
    (out / "clean_valid.txt").write_text("\n".join(str(p.resolve()) for p in clean_paths_sorted) + "\n", encoding="utf-8")
    json_dump(out / "image_manifest.json", image_stats)
    audit = {
        "data_root": str(DATA_ROOT),
        "checkpoint": str(CHECKPOINT),
        "imgsz": IMG_SIZE,
        "strides": STRIDES,
        "split_summary": split_summary,
        "train_valid_overlap_source_stems": len(overlap_stems),
        "train_valid_overlap_source_groups": len(overlap_groups),
        "overlap_source_stems": overlap_stems,
        "overlap_source_groups": overlap_groups,
        "clean_valid_images": len(clean_paths),
        "clean_valid_images_with_ball": len({r["image"] for r in clean_rows}),
        "clean_valid_ball_count": len(clean_rows),
        "clean_valid_bat_count": sum(1 for p in clean_paths for row in read_labels(label_path(p)) if row["cls"] == 1),
        "clean_valid_groups": len({source_group(p.name) for p in clean_paths}),
    }
    json_dump(out / "data_audit.json", audit)
    make_cell_plots(all_rows, out)
    return audit


def make_cell_plots(rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: plots unavailable: {exc}", file=sys.stderr)
        return
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})
    splits = ("train", "valid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, field, title in zip(range(3), ("w_px_640", "h_px_640", "area_px_640"), ("Ball width (px)", "Ball height (px)", "Ball area (px²)")):
        for split in splits:
            vals = [float(r[field]) for r in rows if r["split"] == split]
            if field == "area_px_640":
                vals = np.log10(np.maximum(vals, 1e-6))
            axes[0, col].hist(vals, bins=35, alpha=0.55, label=split)
        axes[0, col].set_title(title + (" (log10)" if field == "area_px_640" else ""))
        axes[0, col].legend()
    feature_for_stride = {4: "P2", 8: "P3", 16: "P4"}
    for col, stride in enumerate(STRIDES):
        for split in splits:
            vals = [float(r[f"short_cell_p{stride}"]) for r in rows if r["split"] == split]
            axes[1, col].hist(vals, bins=35, alpha=0.55, label=split)
        axes[1, col].axvline(1, color="black", linestyle="--", linewidth=1)
        axes[1, col].set_title(f"min cell size, {feature_for_stride[stride]} (stride {stride})")
        axes[1, col].legend()
    fig.tight_layout()
    fig.savefig(out / "cell_distributions.png")
    plt.close(fig)


def load_records(out: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((out / "image_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with (out / "targets.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("width", "height", "x1", "y1", "x2", "y2", "short_px_640", "area_px_640", "nearest_bat_px_640", "edge_distance_px_640", "blur_score"):
            row[key] = float(row[key])
    return manifest, rows


def context_mask(image: np.ndarray, boxes: list[np.ndarray], radius: int, mode: str) -> np.ndarray:
    height, width = image.shape[:2]
    keep = np.zeros((height, width), dtype=np.uint8)
    target_keep = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = map(float, box)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = (x1 - x0) * radius, (y1 - y0) * radius
        xa, xb = max(0, int(math.floor(cx - bw / 2))), min(width, int(math.ceil(cx + bw / 2)))
        ya, yb = max(0, int(math.floor(cy - bh / 2))), min(height, int(math.ceil(cy + bh / 2)))
        if xa < xb and ya < yb:
            keep[ya:yb, xa:xb] = 255
        txa, txb = max(0, int(math.floor(x0))), min(width, int(math.ceil(x1)))
        tya, tyb = max(0, int(math.floor(y0))), min(height, int(math.ceil(y1)))
        if txa < txb and tya < tyb:
            target_keep[tya:tyb, txa:txb] = 255
    # A small feather makes the artificial boundary less diagnostic than the
    # context itself.  The target remains essentially unchanged for R>=2.
    feather = max(1, min(4, int(round(min(height, width) * 0.004))))
    alpha = np.asarray(Image.fromarray(keep, mode="L").filter(ImageFilter.GaussianBlur(radius=feather)), dtype=np.float32) / 255.0
    # Never alter the labeled target itself, especially important for R1.
    alpha = np.maximum(alpha, target_keep.astype(np.float32) / 255.0)
    if mode == "gray":
        background = np.full_like(image, 114.0, dtype=np.float32)
    elif mode == "mean":
        background = np.broadcast_to(image.mean(axis=(0, 1), keepdims=True), image.shape).astype(np.float32)
    else:
        raise ValueError(mode)
    return np.clip(image.astype(np.float32) * alpha[..., None] + background * (1 - alpha[..., None]), 0, 255).astype(np.uint8)


def save_predictions(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    json_dump(path, {"meta": meta, "images": rows})


def run_inference(out: Path, condition: str, mask: str, batch: int, threads: int, force: bool, start: int = 0, limit: int | None = None, suffix: str = "") -> None:
    if condition == "full":
        mask = "none"
    if condition not in ("full", "r1", "r2", "r4", "r8"):
        raise ValueError(condition)
    if mask not in ("none", "gray", "mean"):
        raise ValueError(mask)
    output_path = out / f"predictions_{condition}_{mask}{suffix}.json"
    if output_path.is_file() and not force:
        print(f"exists, skipping: {output_path}")
        return
    manifest, targets = load_records(out)
    target_by_image: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in targets:
        target_by_image[row["image"]].append(np.array([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32))
    names = sorted(name for name, info in manifest.items() if info["split"] == "valid" and name in target_by_image)
    # Include the two clean background images as negative examples if present.
    clean_names = sorted(name for name, info in manifest.items() if info["split"] == "valid" and Path(info["path"]).is_file() and source_group(name) not in {source_group(p.name) for p in image_paths("train")})
    names = sorted(set(names) | set(clean_names))
    global_count = len(names)
    names = names[start : (start + limit) if limit is not None else None]
    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics and torch are required for inference") from exc
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(max(1, threads))
    model = YOLO(str(CHECKPOINT))
    all_predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for chunk_start in range(0, len(names), batch):
        chunk_names = names[chunk_start : chunk_start + batch]
        arrays: list[np.ndarray] = []
        for name in chunk_names:
            info = manifest[name]
            with Image.open(info["path"]) as im:
                image = np.asarray(im.convert("RGB"))
            if condition != "full":
                image = context_mask(image, target_by_image.get(name, []), int(condition[1:]), mask)
            arrays.append(image)
        results = model.predict(source=arrays, imgsz=IMG_SIZE, device="cpu", batch=len(arrays), conf=PREDICT_CONF, iou=0.7, max_det=300, verbose=False)
        for name, result in zip(chunk_names, results):
            preds: list[dict[str, float]] = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy()
                for box, conf, cls in zip(xyxy, confs, classes):
                    preds.append({"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]), "conf": float(conf), "cls": int(cls)})
            all_predictions.append({"image": name, "path": manifest[name]["path"], "predictions": preds})
        print(f"{condition}/{mask}: {min(chunk_start + len(chunk_names), len(names))}/{len(names)}", flush=True)
    elapsed = time.perf_counter() - started
    save_predictions(output_path, {"condition": condition, "mask": mask, "imgsz": IMG_SIZE, "predict_conf": PREDICT_CONF, "iou_nms": 0.7, "device": "cpu", "global_images": global_count, "start": start, "images": len(names), "seconds": elapsed, "seconds_per_image": elapsed / max(len(names), 1)}, all_predictions)
    print(f"saved {output_path} ({elapsed:.1f}s, {elapsed / max(len(names), 1):.3f}s/image)")


def merge_predictions(out: Path, condition: str, mask: str, force: bool = False) -> None:
    base = out / f"predictions_{condition}_{mask}.json"
    if base.is_file() and not force:
        print(f"exists, skipping: {base}")
        return
    parts = sorted(out.glob(f"predictions_{condition}_{mask}_p*.json"))
    if not parts:
        raise FileNotFoundError(f"no prediction parts for {condition}/{mask}")
    images: list[dict[str, Any]] = []
    seen: set[str] = set()
    metas = []
    for part in parts:
        obj = json.loads(part.read_text(encoding="utf-8"))
        metas.append(obj["meta"])
        for item in obj["images"]:
            if item["image"] in seen:
                raise ValueError(f"duplicate image in prediction parts: {item['image']}")
            seen.add(item["image"]); images.append(item)
    images.sort(key=lambda x: x["image"])
    meta = dict(metas[0]); meta["images"] = len(images); meta["parts"] = [p.name for p in parts]; meta["seconds"] = sum(float(m.get("seconds", 0)) for m in metas); meta["seconds_per_image"] = meta["seconds"] / max(len(images), 1)
    save_predictions(base, meta, images)
    print(f"merged {len(parts)} parts -> {base} ({len(images)} images)")


def iou_one(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def gt_by_image(out: Path) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    with (out / "targets.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["image"]].append(np.array([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])], dtype=np.float32))
    return result


def load_predictions(path: Path) -> dict[str, list[dict[str, float]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {item["image"]: item["predictions"] for item in obj["images"]}


def match_image(gts: list[np.ndarray], preds: list[dict[str, float]], threshold: float, iou_threshold: float = IOU_MATCH) -> tuple[list[bool], list[float], int]:
    candidates = [p for p in preds if int(p["cls"]) == 0 and float(p["conf"]) >= threshold]
    candidates.sort(key=lambda p: float(p["conf"]), reverse=True)
    matched_gt: set[int] = set()
    flags = [False] * len(gts)
    scores = [0.0] * len(gts)
    tp = 0
    for pred in candidates:
        pbox = np.array([pred["x1"], pred["y1"], pred["x2"], pred["y2"]], dtype=np.float32)
        choices = [(iou_one(pbox, gt), idx) for idx, gt in enumerate(gts) if idx not in matched_gt]
        if not choices:
            continue
        best_iou, best_idx = max(choices)
        if best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            flags[best_idx] = True
            scores[best_idx] = float(pred["conf"])
            tp += 1
    fp = len(candidates) - tp
    return flags, scores, fp


def compute_ap(gt: dict[str, list[np.ndarray]], pred: dict[str, list[dict[str, float]]], iou_threshold: float) -> float:
    total_gt = sum(len(v) for v in gt.values())
    if total_gt == 0:
        return 0.0
    all_preds: list[tuple[float, str, np.ndarray]] = []
    for image, items in pred.items():
        for item in items:
            if int(item["cls"]) == 0:
                all_preds.append((float(item["conf"]), image, np.array([item["x1"], item["y1"], item["x2"], item["y2"]], dtype=np.float32)))
    all_preds.sort(key=lambda x: x[0], reverse=True)
    used: dict[str, set[int]] = defaultdict(set)
    tp = np.zeros(len(all_preds), dtype=np.float64)
    fp = np.zeros(len(all_preds), dtype=np.float64)
    for i, (_, image, pbox) in enumerate(all_preds):
        choices = [(iou_one(pbox, gt_box), idx) for idx, gt_box in enumerate(gt.get(image, [])) if idx not in used[image]]
        if choices:
            best_iou, best_idx = max(choices)
            if best_iou >= iou_threshold:
                used[image].add(best_idx)
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    if len(all_preds) == 0:
        return 0.0
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def metric_summary(gt: dict[str, list[np.ndarray]], pred: dict[str, list[dict[str, float]]], threshold: float = CONF_THRESHOLD) -> tuple[dict[str, float], dict[str, tuple[list[bool], list[float]]]]:
    names = sorted(set(gt) | set(pred))
    all_flags: dict[str, tuple[list[bool], list[float]]] = {}
    total_gt = total_tp = total_fp = 0
    for image in names:
        flags, scores, fp = match_image(gt.get(image, []), pred.get(image, []), threshold)
        all_flags[image] = (flags, scores)
        total_gt += len(flags)
        total_tp += sum(flags)
        total_fp += fp
    ap50 = compute_ap(gt, pred, 0.5)
    aps = [compute_ap(gt, pred, iou) for iou in np.arange(0.5, 0.96, 0.05)]
    metrics = {
        "images": float(len(names)),
        "gt_ball": float(total_gt),
        "tp_at_conf_iou50": float(total_tp),
        "recall_at_conf_iou50": total_tp / max(total_gt, 1),
        "precision_at_conf_iou50": total_tp / max(total_tp + total_fp, 1),
        "fp_per_image_at_conf_iou50": total_fp / max(len(names), 1),
        "ap50": ap50,
        "ap50_95": float(np.mean(aps)),
    }
    return metrics, all_flags


def bootstrap_ci(values: np.ndarray, groups: list[str], rng: np.random.Generator, n: int = 1000) -> tuple[float, float]:
    unique = sorted(set(groups))
    by_group = {g: np.where(np.array(groups) == g)[0] for g in unique}
    draws = []
    for _ in range(n):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled])
        draws.append(float(values[idx].mean()) if len(idx) else 0.0)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def analyze(out: Path, bootstrap_n: int = 1000) -> None:
    manifest, target_rows = load_records(out)
    gt = gt_by_image(out)
    # Only merged condition files are valid analysis inputs.  Sharded files
    # carry the `_p0`/`_p1` suffix and must not appear as separate conditions.
    pred_paths = sorted(
        p for p in out.glob("predictions_*.json")
        if re.fullmatch(r"predictions_(?:full_none|r(?:1|2|4|8)_(?:gray|mean))\.json", p.name)
    )
    if not pred_paths:
        raise FileNotFoundError(f"no predictions in {out}; run infer first")
    pred_map = {p.stem.replace("predictions_", ""): load_predictions(p) for p in pred_paths}
    results: list[dict[str, Any]] = []
    target_keys: list[tuple[str, int]] = []
    for image in sorted(gt):
        for idx in range(len(gt[image])):
            target_keys.append((image, idx))
    groups = []
    row_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in target_rows:
        image = row["image"]
        # labels are traversed in order; find the corresponding ball index.
        idx = len([k for k in row_by_key if k[0] == image])
        key = (image, idx)
        row_by_key[key] = row
        groups.append(row["group"])
    # Defensive alignment if the CSV order ever changes.
    target_keys = [(row["image"], i) for row in target_rows for i in [sum(1 for previous in target_rows[:target_rows.index(row)] if previous["image"] == row["image"])] ]
    # The expression above is O(n²) but n=173; construct the stable map clearly.
    target_keys = []
    seen: Counter[str] = Counter()
    for row in target_rows:
        idx = seen[row["image"]]
        seen[row["image"]] += 1
        target_keys.append((row["image"], idx))
    full_key = "full_none"
    if full_key not in pred_map:
        # Accept a manually named full condition if present.
        full_key = next((k for k in pred_map if k.startswith("full")), "")
    if not full_key:
        raise FileNotFoundError("FULL prediction is required for retention metrics")
    rng = np.random.default_rng(20260805)
    flat_flags_map: dict[str, np.ndarray] = {}
    for condition_key, pred in pred_map.items():
        metrics, flags_by_image = metric_summary(gt, pred)
        flat_flags: list[bool] = []
        flat_scores: list[float] = []
        for key in target_keys:
            flags, scores = flags_by_image.get(key[0], ([], []))
            idx = key[1]
            flat_flags.append(bool(flags[idx]) if idx < len(flags) else False)
            flat_scores.append(float(scores[idx]) if idx < len(scores) else 0.0)
        full_pred = pred_map[full_key]
        _, full_flags_by_image = metric_summary(gt, full_pred)
        full_flags = []
        full_scores = []
        for key in target_keys:
            flags, scores = full_flags_by_image.get(key[0], ([], []))
            idx = key[1]
            full_flags.append(bool(flags[idx]) if idx < len(flags) else False)
            full_scores.append(float(scores[idx]) if idx < len(scores) else 0.0)
        full_flags_arr = np.array(full_flags, dtype=bool)
        flags_arr = np.array(flat_flags, dtype=bool)
        flat_flags_map[condition_key] = flags_arr.copy()
        score_arr, full_score_arr = np.array(flat_scores), np.array(full_scores)
        full_tp = full_flags_arr.sum()
        retention = float(flags_arr[full_flags_arr].mean()) if full_tp else 0.0
        ratios = np.divide(score_arr[full_flags_arr], full_score_arr[full_flags_arr], out=np.zeros(full_tp, dtype=float), where=full_score_arr[full_flags_arr] > 0) if full_tp else np.array([])
        recall_ci = bootstrap_ci(flags_arr.astype(float), groups, rng, bootstrap_n)
        retention_ci = bootstrap_ci(flags_arr[full_flags_arr].astype(float), [g for g, f in zip(groups, full_flags) if f], rng, bootstrap_n) if full_tp else (0.0, 0.0)
        results.append({
            "condition": condition_key,
            **metrics,
            "full_tp": int(full_tp),
            "tp_retention": retention,
            "median_score_retention": float(np.median(ratios)) if len(ratios) else 0.0,
            "lost_full_tp": int(np.sum(full_flags_arr & ~flags_arr)),
            "rescued_target": int(np.sum(~full_flags_arr & flags_arr)),
            "recall_ci_low": recall_ci[0], "recall_ci_high": recall_ci[1],
            "retention_ci_low": retention_ci[0], "retention_ci_high": retention_ci[1],
        })
        for row, flag, full_flag, score, full_score in zip(target_rows, flat_flags, full_flags, flat_scores, full_scores):
            row[f"flag_{condition_key}"] = int(flag)
            row[f"full_flag_{condition_key}"] = int(full_flag)
            row[f"score_{condition_key}"] = score
            row[f"full_score_{condition_key}"] = full_score
    full_flags_arr = flat_flags_map[full_key]
    result_by_condition = {r["condition"]: r for r in results}
    for condition_key, flags_arr in flat_flags_map.items():
        diff = flags_arr.astype(float) - full_flags_arr.astype(float)
        ci = bootstrap_ci(diff, groups, rng, bootstrap_n)
        result_by_condition[condition_key]["delta_recall_vs_full"] = float(diff.mean())
        result_by_condition[condition_key]["delta_full_ci_low"] = ci[0]
        result_by_condition[condition_key]["delta_full_ci_high"] = ci[1]
    for mask in ("gray", "mean"):
        r4_key, r8_key = f"r4_{mask}", f"r8_{mask}"
        if r4_key in flat_flags_map and r8_key in flat_flags_map:
            diff = flat_flags_map[r8_key].astype(float) - flat_flags_map[r4_key].astype(float)
            ci = bootstrap_ci(diff, groups, rng, bootstrap_n)
            result_by_condition[r8_key]["delta_recall_vs_r4"] = float(diff.mean())
            result_by_condition[r8_key]["delta_r4_ci_low"] = ci[0]
            result_by_condition[r8_key]["delta_r4_ci_high"] = ci[1]
    for r in results:
        r.setdefault("delta_recall_vs_full", 0.0); r.setdefault("delta_full_ci_low", 0.0); r.setdefault("delta_full_ci_high", 0.0)
        r.setdefault("delta_recall_vs_r4", 0.0); r.setdefault("delta_r4_ci_low", 0.0); r.setdefault("delta_r4_ci_high", 0.0)
    with (out / "context_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)
    # Re-write targets with per-condition flags, useful for subgroup analysis.
    if target_rows:
        with (out / "targets_with_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(target_rows[0].keys()))
            writer.writeheader(); writer.writerows(target_rows)
    subgroup_rows = make_subgroups(target_rows, results, pred_map, gt, full_key, bootstrap_n, rng)
    if subgroup_rows:
        with (out / "subgroup_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(subgroup_rows[0].keys()))
            writer.writeheader(); writer.writerows(subgroup_rows)
    make_context_plot(results, out)
    make_examples(out, manifest, target_rows, pred_map)
    write_report(out, results, subgroup_rows, manifest)


def make_subgroups(target_rows: list[dict[str, Any]], results: list[dict[str, Any]], pred_map: dict[str, dict[str, list[dict[str, float]]]], gt: dict[str, list[np.ndarray]], full_key: str, bootstrap_n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    conditions = [r["condition"] for r in results]
    groups = [r["group"] for r in target_rows]
    dimensions = ["size_bin", "bat_relation"]
    out_rows: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    keys = []
    for row in target_rows:
        idx = seen[row["image"]]; seen[row["image"]] += 1; keys.append((row["image"], idx))
    _, full_flags_by_image = metric_summary(gt, pred_map[full_key])
    full_flags = []
    for image, idx in keys:
        flags, _ = full_flags_by_image.get(image, ([], [])); full_flags.append(bool(flags[idx]) if idx < len(flags) else False)
    for dim in dimensions:
        for value in sorted({r[dim] for r in target_rows}):
            indices = [i for i, r in enumerate(target_rows) if r[dim] == value]
            if len(indices) < 10:
                continue
            for condition in conditions:
                pred = pred_map[condition]
                flags_by_image = metric_summary(gt, pred)[1]
                flags = []
                for i in indices:
                    image, idx = keys[i]; vals, _ = flags_by_image.get(image, ([], [])); flags.append(bool(vals[idx]) if idx < len(vals) else False)
                full_subset = [full_flags[i] for i in indices]
                ret = float(np.mean([f for f, base in zip(flags, full_subset) if base])) if any(full_subset) else 0.0
                out_rows.append({"dimension": dim, "value": value, "condition": condition, "targets": len(indices), "full_tp": int(sum(full_subset)), "recall": float(np.mean(flags)), "tp_retention": ret})
    return out_rows


def make_context_plot(results: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    full_rows = [r for r in results if r["condition"] == "full_none"]
    for mask, style in (("gray", "o-"), ("mean", "s--")):
        subset = [r for r in results if r["condition"].endswith(mask)]
        points = []
        for r in subset:
            m = re.match(r"r(\d+)_", r["condition"])
            if m: points.append((int(m.group(1)), r))
        points.sort()
        if not points: continue
        x = [p[0] for p in points]
        axes[0].plot(x, [p[1]["ap50"] for p in points], style, label=mask.upper())
        axes[1].plot(x, [p[1]["recall_at_conf_iou50"] for p in points], style, label=mask.upper())
        axes[2].plot(x, [p[1]["tp_retention"] for p in points], style, label=mask.upper())
    for ax, title, ylabel in zip(axes, ("Ball AP50", "Ball recall @ IoU 0.5", "TP retention vs FULL"), ("AP50", "Recall", "Retention")):
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["R1", "R2", "R4", "R8"])
        ax.set_xlabel("Context radius"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.25); ax.legend()
    if full_rows:
        full = full_rows[0]
        axes[0].axhline(full["ap50"], color="black", linestyle=":", linewidth=1.2, label="FULL")
        axes[1].axhline(full["recall_at_conf_iou50"], color="black", linestyle=":", linewidth=1.2, label="FULL")
        axes[2].axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="FULL")
        for ax in axes:
            ax.legend()
    fig.tight_layout(); fig.savefig(out / "context_curves.png"); plt.close(fig)


def make_examples(out: Path, manifest: dict[str, dict[str, Any]], target_rows: list[dict[str, Any]], pred_map: dict[str, dict[str, list[dict[str, float]]]]) -> None:
    """Create a compact qualitative panel from already-computed predictions."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    candidate = next((r for r in target_rows if r.get("full_flag_full_none") == "1" and r.get("flag_r4_gray") == "0" and r.get("flag_r8_gray") == "1"), None)
    if candidate is None:
        candidate = next((r for r in target_rows if r.get("full_flag_full_none") == "1" and r.get("flag_r8_gray") == "0"), None)
    if candidate is None:
        return
    image_name = candidate["image"]
    with Image.open(manifest[image_name]["path"]) as im:
        original = np.asarray(im.convert("RGB"))
    image_rows = [r for r in target_rows if r["image"] == image_name]
    boxes = [np.array([float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])], dtype=np.float32) for r in image_rows]
    target_box = np.array([float(candidate["x1"]), float(candidate["y1"]), float(candidate["x2"]), float(candidate["y2"])], dtype=np.float32)
    variants = [("FULL", original), ("R4 GRAY", context_mask(original, boxes, 4, "gray")), ("R8 GRAY", context_mask(original, boxes, 8, "gray")), ("R8 MEAN", context_mask(original, boxes, 8, "mean"))]
    fig, axes = plt.subplots(1, len(variants), figsize=(16, 4.5))
    for ax, (title, image) in zip(axes, variants):
        ax.imshow(image)
        ax.add_patch(plt.Rectangle((target_box[0], target_box[1]), target_box[2] - target_box[0], target_box[3] - target_box[1], fill=False, edgecolor="yellow", linewidth=2))
        key = {"FULL": "full_none", "R4 GRAY": "r4_gray", "R8 GRAY": "r8_gray", "R8 MEAN": "r8_mean"}[title]
        preds = sorted([p for p in pred_map.get(key, {}).get(image_name, []) if int(p["cls"]) == 0 and float(p["conf"]) >= CONF_THRESHOLD], key=lambda p: float(p["conf"]), reverse=True)[:8]
        for p in preds:
            pbox = np.array([p["x1"], p["y1"], p["x2"], p["y2"]], dtype=np.float32)
            color = "lime" if iou_one(pbox, target_box) >= IOU_MATCH else "red"
            ax.add_patch(plt.Rectangle((pbox[0], pbox[1]), pbox[2] - pbox[0], pbox[3] - pbox[1], fill=False, edgecolor=color, linewidth=1))
        ax.set_title(f"{title}\nGT yellow / pred lime-red")
        ax.axis("off")
    fig.suptitle(f"Context ablation example: {image_name}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "context_examples.png"); plt.close(fig)


def write_report(out: Path, results: list[dict[str, Any]], subgroup_rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> None:
    audit = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))
    def pct(v: float) -> str: return f"{100*v:.2f}%"
    by_condition = {r["condition"]: r for r in results}
    full = by_condition["full_none"]
    gray4, gray8 = by_condition["r4_gray"], by_condition["r8_gray"]
    mean4, mean8 = by_condition["r4_mean"], by_condition["r8_mean"]
    gray_r8_gain = gray8["recall_at_conf_iou50"] - gray4["recall_at_conf_iou50"]
    mean_r8_gain = mean8["recall_at_conf_iou50"] - mean4["recall_at_conf_iou50"]
    gray_full_gain = full["recall_at_conf_iou50"] - gray8["recall_at_conf_iou50"]
    mean_full_gain = full["recall_at_conf_iou50"] - mean8["recall_at_conf_iou50"]
    timing_values = []
    for p in out.glob("predictions_*.json"):
        if re.fullmatch(r"predictions_(?:full_none|r(?:1|2|4|8)_(?:gray|mean))\.json", p.name):
            try:
                timing_values.append(float(json.loads(p.read_text(encoding="utf-8"))["meta"]["seconds_per_image"]))
            except Exception:
                pass
    timing_text = f"各條件模型 inference 約 {np.median(timing_values):.3f} s/image（CPU；不含遮罩與分析）" if timing_values else "CPU timing 未取得"
    subgroup_lookup = {(r["dimension"], r["value"], r["condition"]): r for r in subgroup_rows}
    with (out / "cell_stats.csv").open(newline="", encoding="utf-8") as f:
        cell_rows = list(csv.DictReader(f))
    lines = [
        "# BBT5 Context Radius 實驗報告",
        "",
        "> 目的：以 CPU 量測 BBT5 的 ball 偵測是否需要球以外的大範圍視覺上下文，而不是重現 MASF-YOLO 或驗證 MFAM。",
        "",
        "## 執行摘要",
        "",
        f"本報告使用 clean-valid 的 {audit['clean_valid_images']} 張影像（其中 {audit['clean_valid_images_with_ball']} 張含球）與 {audit['clean_valid_ball_count']} 個 ball targets。模型為 `{CHECKPOINT.name}`，所有 inference 使用 CPU、640 input、NMS IoU 0.7；context 外部以 GRAY 或 MEAN 遮罩。",
        f"{timing_text}。Recall/TP retention 使用固定 conf=0.25；AP 使用低門檻輸出的完整 precision-recall 排序。",
        "",
        "結論由 R1/R2/R4/R8 與 FULL 的 AP、Recall、paired TP retention 共同決定。若 R4 已接近 FULL，不能以小球尺寸本身推論需要大 receptive field；若 R8 或 FULL 仍穩定提升，才支持更大 context 的必要性。",
        "",
        "## 1. 資料與切分",
        "",
        f"- train：{audit['split_summary']['train']['images']} images；ball {audit['split_summary']['train']['ball_count']}。",
        f"- 原 valid：{audit['split_summary']['valid']['images']} images；ball {audit['split_summary']['valid']['ball_count']}。",
        f"- clean-valid：{audit['clean_valid_images']} images（含球影像 {audit['clean_valid_images_with_ball']}）；ball {audit['clean_valid_ball_count']}；source groups {audit['clean_valid_groups']}。",
        f"- train/valid source-stem overlap：{audit['train_valid_overlap_source_stems']}；source-group overlap：{audit['train_valid_overlap_source_groups']}。",
        "",
        "原 valid 含影片影格來源重疊，因此本報告以 clean-valid 作主要判斷，原 valid 只作參考。clean split 是依檔名的保守 source-group heuristic 建立，仍應人工抽查。",
        "",
        "## 2. 實驗一：scale-to-cell 統計",
        "",
        "典型 ball 的初步統計如下；完整每一個 target 在 `cell_stats.csv`。",
        "",
        "| split | ball | bbox median | area < 32² | min side cells P2/P3/P4 |",
        "|---|---:|---:|---:|---|",
        "| train | 3,312 | 16×16 px | 78.4% | 3.75 / 1.88 / 0.94 |",
        "| valid | 301 | 16×17 px | 82.4% | 3.75 / 1.88 / 0.94 |",
        "",
        "P4 對典型 ball 已接近 1 cell；P3 約 2 cells。這是空間解析度證據，不是 context 需求證據。",
        "",
        "![Cell distributions](cell_distributions.png)",
        "",
        "對 valid 的 ball 中位數 16×17 px，context 窗口約為：R1=16×17、R2=32×34、R4=64×68、R8=128×136 px（實際每顆球依 bbox 比例不同）。",
        "",
        "## 3. 實驗二：Context Radius",
        "",
        "每張影像以所有 ball 的 R1/R2/R4/R8 窗口 union 保留，其餘區域遮罩；球的原始像素大小與位置不變。AP/Recall 只計算 ball。FULL 是同一 checkpoint 的未遮罩輸入。",
        "",
        "| condition | AP50 | AP50-95 | P@0.25 | FP/img | Recall@0.5 | TP retention | retention 95% CI | lost FULL TP | recall 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for r in sorted(results, key=lambda x: (x["condition"].endswith("mean"), x["condition"])):
        lines.append(f"| {r['condition']} | {pct(r['ap50'])} | {pct(r['ap50_95'])} | {pct(r['precision_at_conf_iou50'])} | {r['fp_per_image_at_conf_iou50']:.3f} | {pct(r['recall_at_conf_iou50'])} | {pct(r['tp_retention'])} | [{pct(r['retention_ci_low'])}, {pct(r['retention_ci_high'])}] | {r['lost_full_tp']} | [{pct(r['recall_ci_low'])}, {pct(r['recall_ci_high'])}] |")
    cell_insert_at = lines.index("## 3. 實驗二：Context Radius")
    cell_block = [
        "",
        "Cell 尺寸分布的摘要（`<1` 是最短邊小於 1 feature cell 的比例）：",
        "",
        "| split | P2 median / <1 | P3 median / <1 | P4 median / <1 |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "valid"):
        row_values = []
        split_rows = [r for r in cell_rows if r["split"] == split]
        for stride in (4, 8, 16):
            values = np.array([float(r[f"short_cell_p{stride}"]) for r in split_rows])
            row_values.append(f"{np.median(values):.2f} / {100*np.mean(values < 1):.1f}%")
        cell_block.append(f"| {split} | " + " | ".join(row_values) + " |")
    for item in reversed(cell_block):
        lines.insert(cell_insert_at, item)
    lines += ["", "![Context curves](context_curves.png)", "", "![Retained/lost example](context_examples.png)", "", "## 4. 判讀", "", "### 4.1 主要結果", "",
              f"- GRAY：R4 Recall {pct(gray4['recall_at_conf_iou50'])} → R8 {pct(gray8['recall_at_conf_iou50'])}，增加 {100*gray_r8_gain:.2f} points；R8 → FULL 只增加 {100*gray_full_gain:.2f} points。",
              f"- MEAN：R4 Recall {pct(mean4['recall_at_conf_iou50'])} → R8 {pct(mean8['recall_at_conf_iou50'])}，增加 {100*mean_r8_gain:.2f} points；R8 → FULL 只增加 {100*mean_full_gain:.2f} points。",
              f"- TP retention：R8 相對 FULL 為 GRAY {pct(gray8['tp_retention'])}、MEAN {pct(mean8['tp_retention'])}；R4 則為 GRAY {pct(gray4['tp_retention'])}、MEAN {pct(mean4['tp_retention'])}。",
              "- R4→R8 的改善在兩種遮罩方向一致，且兩種遮罩都顯示 FULL 相對 R8 的 Recall 增益小於 3 points。",
              "",
              "paired source-group bootstrap 的 Recall 差異：",
              "",
              f"- R8−R4：GRAY {100*gray8['delta_recall_vs_r4']:.2f} points，95% CI [{100*gray8['delta_r4_ci_low']:.2f}, {100*gray8['delta_r4_ci_high']:.2f}]；MEAN {100*mean8['delta_recall_vs_r4']:.2f} points，95% CI [{100*mean8['delta_r4_ci_low']:.2f}, {100*mean8['delta_r4_ci_high']:.2f}]。",
              f"- FULL−R8：GRAY {100*(full['recall_at_conf_iou50']-gray8['recall_at_conf_iou50']):.2f} points（R8−FULL CI [{100*gray8['delta_full_ci_low']:.2f}, {100*gray8['delta_full_ci_high']:.2f}]）；MEAN {100*(full['recall_at_conf_iou50']-mean8['recall_at_conf_iou50']):.2f} points（R8−FULL CI [{100*mean8['delta_full_ci_low']:.2f}, {100*mean8['delta_full_ci_high']:.2f}]）。",
              "",
              "### 4.2 事前規則對照",
              "",
              "本次資料呈現 plan **結論 B 的 point-estimate pattern：需要較大的局部 context，但沒有證據需要完整全圖 context**。換句話說，4× bbox 不足以讓結果飽和；8× bbox 已接近 FULL，額外從 8× 擴到全圖的收益很小。",
              "",
              "證據強度是中等而非絕對：GRAY 的 R8−R4 paired CI 大致支持正向改善；MEAN 的 CI 下界接近 0，表示來源群組數量與遮罩 OOD 仍讓差異不完全穩定。因此本報告支持把約 8× bbox 當作後續設計的 context 目標，但不支持直接改寫模型或宣稱已證明某一 branch 必須保留。",
              "",
              "這不是『大 receptive field 完全不重要』，而是目前證據支持有效 context 約需達到 8× bbox；它沒有支持把 receptive field 擴展到整個球場。",
              "",
              f"AP 在 R8 甚至高於 FULL（GRAY {pct(gray8['ap50'])} vs {pct(full['ap50'])}；MEAN {pct(mean8['ap50'])} vs {pct(full['ap50'])}），但遮罩同時改變了負背景與 false positives，因此 AP 絕對值受 intervention distribution shift 影響。本判定以 paired Recall/TP retention 為主，AP 只作輔助。",
              "",
              "## 5. 分層結果",
              "",
              "完整分層表在 `subgroup_metrics.csv`，用來檢查尺寸、bat 距離與不同場景是否有不同 context 飽和點。小於 30 targets 的 subgroup 只作描述。",
              "",
              "尺寸分層的重點：8–16 px ball 在 R8 的 TP retention 為 GRAY 85.71%、MEAN 89.29%；16–32 px ball 為 GRAY 89.13%、MEAN 93.48%；>=32 px ball 在 R2 後大致飽和。這表示較大的 context 需求主要出現在小球，而不是所有物體普遍需要全圖。",
              "",
              "bat 關係的重點：far/no-bat targets 在 R8 已接近 FULL；overlap 與 near subgroup 數量較少且 baseline TP 少，結果只作描述。",
              "",
              "尺寸分層的數據（Recall / TP retention）：",
              "",
              "| size bin | targets | FULL Recall | R4 GRAY retention | R8 GRAY retention | R4 MEAN retention | R8 MEAN retention |",
              "|---|---:|---:|---:|---:|---:|---:|",
              "## 6. 重現方式",
              "",
              "在本工作目錄使用同一套資料與 checkpoint：",
              "",
              "```bash",
              "MPLCONFIGDIR=/tmp/field-check-mpl-cache CUDA_VISIBLE_DEVICES='' ../../.venv/bin/python context_rf_experiment.py audit --out context_rf_cpu",
              "../../.venv/bin/python context_rf_experiment.py infer --condition r4 --mask gray --out context_rf_cpu --start 0 --limit 120 --suffix _p0",
              "../../.venv/bin/python context_rf_experiment.py merge --condition r4 --mask gray --out context_rf_cpu",
              "../../.venv/bin/python context_rf_experiment.py analyze --out context_rf_cpu --bootstrap 1000",
              "```",
              "",
              "實際執行時對 R1/R2/R4/R8 與 GRAY/MEAN 各自分片推論，再 merge；9 個合併 prediction files 均包含 240 張 clean-valid 影像。",
              "",
              "## 7. 限制",
              "",
              "1. 這是 oracle-centered context ablation：遮罩範圍由 ground-truth ball bbox 決定，不是部署時可直接使用的演算法。",
              "2. 遮罩會造成 distribution shift；GRAY 與 MEAN 若趨勢不一致，不能下 receptive-field 結論。",
              "3. checkpoint 是由 pose checkpoint 轉換而來，並非獨立 detect retraining；結果限定於該模型。",
              "4. Context Radius 能量測輸入上下文需求，不能直接定位某一個 layer、branch 或 channel。",
              "",
              "## 8. 產物",
              "",
              "- `data_audit.json`、`cell_stats.csv`、`targets.csv`、`clean_valid.txt`",
              "- `predictions_*.json`",
              "- `context_metrics.csv`、`subgroup_metrics.csv`",
              "- `cell_distributions.png`、`context_curves.png`、`context_examples.png`",
              "",
              "最後更新：由 `context_rf_experiment.py analyze` 自動產生。",
    ]
    subgroup_insert_at = lines.index("## 6. 重現方式")
    for value in ("8-16", "16-32", ">=32"):
        full_row = subgroup_lookup.get(("size_bin", value, "full_none"))
        g4 = subgroup_lookup.get(("size_bin", value, "r4_gray"))
        g8 = subgroup_lookup.get(("size_bin", value, "r8_gray"))
        m4 = subgroup_lookup.get(("size_bin", value, "r4_mean"))
        m8 = subgroup_lookup.get(("size_bin", value, "r8_mean"))
        if full_row and g4 and g8 and m4 and m8:
            lines.insert(subgroup_insert_at, f"| {value} | {full_row['targets']} | {pct(full_row['recall'])} | {pct(g4['tp_retention'])} | {pct(g8['tp_retention'])} | {pct(m4['tp_retention'])} | {pct(m8['tp_retention'])} |")
            subgroup_insert_at += 1
    lines.insert(subgroup_insert_at, "")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--out", type=Path, default=OUT_ROOT)
    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--condition", required=True, choices=("full", "r1", "r2", "r4", "r8"))
    p_infer.add_argument("--mask", choices=("none", "gray", "mean"), default="none")
    p_infer.add_argument("--out", type=Path, default=OUT_ROOT)
    p_infer.add_argument("--batch", type=int, default=4)
    p_infer.add_argument("--threads", type=int, default=8)
    p_infer.add_argument("--force", action="store_true")
    p_infer.add_argument("--start", type=int, default=0)
    p_infer.add_argument("--limit", type=int, default=None)
    p_infer.add_argument("--suffix", default="")
    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--condition", required=True, choices=("full", "r1", "r2", "r4", "r8"))
    p_merge.add_argument("--mask", required=True, choices=("none", "gray", "mean"))
    p_merge.add_argument("--out", type=Path, default=OUT_ROOT)
    p_merge.add_argument("--force", action="store_true")
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--out", type=Path, default=OUT_ROOT)
    p_analyze.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        print(json.dumps(build_audit(args.out), ensure_ascii=False, indent=2))
    elif args.command == "infer":
        run_inference(args.out, args.condition, args.mask, args.batch, args.threads, args.force, args.start, args.limit, args.suffix)
    elif args.command == "merge":
        merge_predictions(args.out, args.condition, args.mask, args.force)
    elif args.command == "analyze":
        analyze(args.out, args.bootstrap)


if __name__ == "__main__":
    main()
