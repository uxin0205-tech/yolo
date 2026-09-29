#!/usr/bin/env python3
"""由既有 predictions.json 計算 canonical COCO API 整體與 ball／bat AP。"""

from __future__ import annotations

import argparse
import contextlib
import io
from datetime import datetime, timezone
from pathlib import Path

from _bundle import atomic_json, file_sha256

SCOPES = {
    "overall": None,
    "sports_ball": [37],
    "baseball_bat": [39],
}


def evaluate(annotations: Path, predictions: Path, category_ids: list[int] | None) -> dict[str, float | int]:
    """執行 COCOeval；category id 使用 COCO91 原始 ID。"""

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotations))
        detections = ground_truth.loadRes(str(predictions))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        if category_ids is not None:
            evaluator.params.catIds = category_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    annotations_for_scope = [
        item
        for item in ground_truth.dataset["annotations"]
        if category_ids is None or int(item["category_id"]) in category_ids
    ]
    image_ids = {int(item["image_id"]) for item in annotations_for_scope}
    return {
        "images": len(ground_truth.imgs) if category_ids is None else len(image_ids),
        "instances": len(annotations_for_scope),
        "ap50_95": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "ap_s": float(evaluator.stats[3]),
        "ap_m": float(evaluator.stats[4]),
        "ap_l": float(evaluator.stats[5]),
        "ar100": float(evaluator.stats[8]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id")
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "evaluator": "pycocotools COCOeval bbox",
        "annotations": {
            "path": str(args.annotations.resolve()),
            "sha256": file_sha256(args.annotations),
        },
        "predictions": {
            "path": str(args.predictions.resolve()),
            "sha256": file_sha256(args.predictions),
        },
        "category_contract": {
            "sports_ball": {"ultralytics_index": 32, "coco_category_id": 37},
            "baseball_bat": {"ultralytics_index": 34, "coco_category_id": 39},
        },
        "metrics": {
            name: evaluate(args.annotations, args.predictions, category_ids)
            for name, category_ids in SCOPES.items()
        },
    }
    atomic_json(args.output, payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
