from __future__ import annotations

from masf_yolo.evaluation.runner import build_parser


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
    assert args.device == 0
