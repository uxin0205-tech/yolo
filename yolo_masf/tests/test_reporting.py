from __future__ import annotations

import json
from pathlib import Path

from masf_yolo.reporting import rebuild_report


def _b0_metrics(*, offset: float) -> dict[str, object]:
    return {
        "map50_95": 0.50 + offset,
        "per_class": {
            "ball": {"ap": 0.51 + offset, "ap50": 0.61 + offset},
            "bat": {"ap": 0.52 + offset, "ap50": 0.62 + offset},
        },
        "class_diagnostics": {
            "ball": {
                "gt_count": 12,
                "prediction_count": 10,
                "true_positive_count": 8,
                "missed_count": 4,
                "false_positive_count": 2,
                "precision": 0.8,
                "recall": 2 / 3,
                "subsets": {
                    "tiny": {"gt_count": 3, "recall": 1 / 3},
                    "small": {"gt_count": 5, "recall": 0.8},
                    "large": {"gt_count": 4, "recall": 0.75},
                    "blur_proxy": {"gt_count": 2, "recall": 0.5},
                },
            },
            "bat": {
                "gt_count": 7,
                "prediction_count": 8,
                "true_positive_count": 6,
                "missed_count": 1,
                "false_positive_count": 2,
                "precision": 0.75,
                "recall": 6 / 7,
                "subsets": {
                    "tiny": {"gt_count": 0, "recall": None},
                    "small": {"gt_count": 1, "recall": 1.0},
                    "large": {"gt_count": 6, "recall": 5 / 6},
                    "blur_proxy": {"gt_count": 3, "recall": 2 / 3},
                },
            },
        },
    }


def test_report_rebuilds_from_available_immutable_artifacts(tmp_path: Path) -> None:
    work = tmp_path / "work"
    config_dir = work / "configs"
    config_dir.mkdir(parents=True)
    config = config_dir / "static-phase1.yaml"
    config.write_text(
        """
schema_version: 1
pipeline_name: static-phase1
artifacts_root: artifacts/static-phase1
variants: [B1, M0, M1, M2, M3]
environment: {python: '3.14', torch: '2.11.0+cu128', cuda: '12.8', ultralytics: '8.4.90', faster_coco_eval: '1.7.2'}
dataset: {source: dataset, split_ratios: [0.8, 0.1, 0.1], class_names: [ball, bat], seed: 42, minimum_ball_count: 50}
model: {source_weights: weights.pt, source_weights_sha256: a, official_model_yaml_sha256: b, official_default_yaml_sha256: c, imgsz: 640, nc: 2, scale: m}
training: {optimizer: SGD, momentum: 0.937, cos_lr: true, deterministic: true, amp: true, nbs: 64, batch_candidates: [16, 8, 4, 2, 1], b1_a_epochs: 10, b1_b_epochs: 90, smoke_epochs: 3, variant_epochs: 100, b1_a_lr0: 0.01, formal_lr0: 0.001, freeze: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
pipeline: {max_attempts: 3, device: 0, systemd_unit_prefix: masf-yolo-phase1}
""".strip()
    )
    artifacts = work / "artifacts" / "static-phase1"
    (artifacts / "dataset").mkdir(parents=True)
    (artifacts / "dataset" / "manifest.json").write_text(json.dumps({"dataset_hash": "data-hash"}))
    (artifacts / "selection.json").write_text(json.dumps({"selected": "M3", "reason": "efficiency_equivalent"}))
    (artifacts / "final_audit.json").write_text(json.dumps({"ok": True, "errors": []}))
    reference = artifacts / "references"
    reference.mkdir()
    (reference / "b0.json").write_text(
        json.dumps(
            {
                "checkpoint_hash": "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d",
                "provenance": "BBT5 pose checkpoint converted to detect",
                "data_exposed": True,
                "selection_eligible": False,
            }
        )
    )
    for split, offset in (("val", 0.0), ("test", 0.1)):
        evaluation = artifacts / "evaluation" / split / "b0"
        evaluation.mkdir(parents=True)
        (evaluation / "metrics.json").write_text(json.dumps(_b0_metrics(offset=offset)))

    report_path = Path(rebuild_report(config))
    text = report_path.read_text()

    assert "BEST_PARTIAL: M3" in text
    assert "data-hash" in text
    assert "Final audit: PASS" in text
    assert "B0 is pose-derived and data-exposed" in text
    assert "formal_m7: pending" in text
    assert "- B0:" in text
    assert "- M7:" in text
    assert "### B0" in text
    assert "Ball: AP50=" in text
    assert "Bat: AP50=" in text
    assert "GT=12" in text
    assert "predictions=10" in text
    assert "missed=4" in text
    assert "false positives=2" in text
