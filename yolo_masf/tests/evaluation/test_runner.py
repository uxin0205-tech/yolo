from __future__ import annotations

from pathlib import Path

from masf_yolo.evaluation.runner import assemble_evaluation_results, build_parser


def test_single_model_evaluation_cli_parses_explicit_artifacts() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "b0.pt",
            "--data",
            "data.yaml",
            "--coco",
            "val.coco.json",
            "--split",
            "val",
            "--output",
            "evaluation/val/b0",
            "--device",
            "0",
        ]
    )

    assert args.checkpoint.name == "b0.pt"
    assert args.split == "val"
    assert args.device == "0"


def test_single_model_evaluation_cli_accepts_cpu_device() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "b0.pt",
            "--data",
            "data.yaml",
            "--coco",
            "val.coco.json",
            "--split",
            "val",
            "--output",
            "evaluation/val/b0-cpu",
            "--device",
            "cpu",
        ]
    )

    assert args.device == "cpu"


def test_evaluation_results_expose_both_classes_and_keep_ball_selection_fields() -> None:
    ground_truth = {
        "images": [{"id": 1}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 6, 6]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 30, 10]},
        ],
    }
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 6, 6], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [20, 20, 30, 10], "score": 0.8},
    ]
    coco_metrics = {
        "map50_95": 0.55,
        "per_class": {
            "ball": {"ap": 0.4, "ap_s": 0.3},
            "bat": {"ap": 0.7, "ap_s": None},
        },
    }

    results = assemble_evaluation_results(
        checkpoint=Path("model.pt"),
        split="val",
        ground_truth=ground_truth,
        predictions=predictions,
        coco_metrics=coco_metrics,
        ultralytics_results={"metrics/mAP50-95(B)": 0.55},
    )

    assert set(results["class_diagnostics"]) == {"ball", "bat"}
    assert results["class_diagnostics"]["ball"]["recall"] == 1.0
    assert results["class_diagnostics"]["bat"]["precision"] == 1.0
    assert results["ball_recall"] == 1.0
    assert results["ball_gt_count"] == 1
    assert results["ball_ap"] == 0.4
    assert results["ball_ap_s"] == 0.3
    assert results["ball_subsets"] == results["class_diagnostics"]["ball"]["subsets"]
