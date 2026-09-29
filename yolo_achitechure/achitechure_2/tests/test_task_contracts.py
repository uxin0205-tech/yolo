from __future__ import annotations

from achitechure_2.cpu_validation import validate_cpu_candidate


def test_cpu_dry_run_covers_tasks_loss_gradients_640_and_reload(combined_parent) -> None:
    report = validate_cpu_candidate(
        combined_parent,
        builder=type(combined_parent),
        frozen_module_paths=("trunk.layers.2",),
        smoke_imgsz=32,
        geometry_imgsz=640,
    )

    assert report.passed
    assert report.device == "cpu"
    assert report.tasks == ("detect", "pose", "both")
    assert report.output_shapes["detect"]["detect"] == (1, 80, 32, 32)
    assert report.output_shapes["pose"]["pose"] == (1, 8, 32, 32)
    assert set(report.output_shapes["both"]) == {"detect", "pose"}
    assert report.geometry_shapes["detect"] == (1, 80, 640, 640)
    assert report.geometry_shapes["pose"] == (1, 8, 640, 640)
    assert report.loss_is_finite
    assert report.gradients_are_finite
    assert report.trainable_gradient_count > 0
    assert report.frozen_gradient_count == 0
    assert report.state_dict_reload
    assert report.contract_unchanged


def test_cpu_dry_run_does_not_claim_accuracy_or_latency(combined_parent) -> None:
    report = validate_cpu_candidate(
        combined_parent,
        builder=type(combined_parent),
        frozen_module_paths=("trunk.layers.2",),
        smoke_imgsz=16,
        geometry_imgsz=32,
    )

    payload = report.to_dict()
    assert "latency" not in payload
    assert "map" not in payload
    assert payload["scope"] == "CPU 結構／數值 smoke；不代表準確度或正式效能"
