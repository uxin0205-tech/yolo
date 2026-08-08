"""Ultralytics inference plus faster-coco-eval in locked letterbox space."""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
from typing import Any

import faster_coco_eval
from faster_coco_eval import COCO, COCOeval_faster

from masf_yolo.artifacts.io import atomic_write_json

from .metrics import (
    ball_observations_from_coco,
    summarize_ball_subsets,
    translate_predictions_to_letterbox,
)
from .galleries import write_false_positive_gallery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    return parser


def _run_evaluator(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    category_ids: list[int] | None = None,
) -> list[float]:
    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = copy.deepcopy(ground_truth)
        coco_dt.dataset["annotations"] = []
        coco_dt.createIndex()
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    if category_ids is not None:
        evaluator.params.catIds = category_ids
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return [float(value) if value >= 0 else None for value in evaluator.stats]  # type: ignore[list-item]


def evaluate_coco(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    stats = _run_evaluator(ground_truth, predictions)
    categories = {int(category["id"]): category["name"] for category in ground_truth["categories"]}
    per_class: dict[str, dict[str, float | int | None]] = {}
    for category_id, name in categories.items():
        class_stats = _run_evaluator(ground_truth, predictions, [category_id])
        per_class[name] = {
            "ap": class_stats[0],
            "ap50": class_stats[1],
            "ap75": class_stats[2],
            "ap_s": class_stats[3],
            "gt_count": sum(
                int(annotation["category_id"]) == category_id
                for annotation in ground_truth["annotations"]
            ),
        }
    return {
        "evaluator": f"faster-coco-eval {faster_coco_eval.__version__}",
        "map50_95": stats[0],
        "map50": stats[1],
        "map75": stats[2],
        "ap_s": stats[3],
        "ap_m": stats[4],
        "ap_l": stats[5],
        "per_class": per_class,
    }


def run_variant_evaluation(
    checkpoint: Path,
    data_yaml: Path,
    coco_path: Path,
    *,
    split: str,
    output_dir: Path,
    device: int = 0,
) -> dict[str, Any]:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint), task="detect")
    ultralytics_metrics = model.val(
        data=str(data_yaml),
        split=split,
        device=device,
        save_json=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        plots=False,
    )
    prediction_path = Path(ultralytics_metrics.save_dir) / "predictions.json"
    raw_predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(coco_path.read_text(encoding="utf-8"))
    predictions = translate_predictions_to_letterbox(raw_predictions, ground_truth)
    coco_metrics = evaluate_coco(ground_truth, predictions)
    observations = ball_observations_from_coco(ground_truth, predictions)
    subsets = summarize_ball_subsets(observations)
    ball_count = len(observations)
    ball_recall = sum(observation.matched for observation in observations) / ball_count if ball_count else None
    results = {
        **coco_metrics,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "ball_recall": ball_recall,
        "ball_gt_count": ball_count,
        "ball_ap": coco_metrics["per_class"]["ball"]["ap"],
        "ball_ap_s": coco_metrics["per_class"]["ball"]["ap_s"],
        "ball_subsets": subsets,
        "ultralytics": {
            key: float(value) for key, value in ultralytics_metrics.results_dict.items()
        },
    }
    atomic_write_json(output_dir / "predictions.letterbox.json", predictions)
    atomic_write_json(output_dir / "metrics.json", results)
    write_false_positive_gallery(ground_truth, predictions, output_dir)
    return results


def main() -> None:
    args = build_parser().parse_args()
    results = run_variant_evaluation(
        args.checkpoint,
        args.data,
        args.coco,
        split=args.split,
        output_dir=args.output,
        device=args.device,
    )
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
