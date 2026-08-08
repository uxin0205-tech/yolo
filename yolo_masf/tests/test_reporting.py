from __future__ import annotations

import json
from pathlib import Path

from masf_yolo.reporting import rebuild_report


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

    report_path = Path(rebuild_report(config))
    text = report_path.read_text()

    assert "BEST_PARTIAL: M3" in text
    assert "data-hash" in text
    assert "Final audit: PASS" in text
    assert "B0 is pose-derived and data-exposed" in text
    assert "formal_m7: pending" in text
    assert "- B0:" in text
    assert "- M7:" in text
