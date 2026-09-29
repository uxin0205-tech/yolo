from __future__ import annotations

from pathlib import Path

from masf_yolo.runtime import (
    GpuSnapshot,
    pipeline_identity,
    wait_for_gpu_idle,
    wait_for_predecessor_units,
    verify_environment,
)


def test_environment_verification_matches_all_pinned_local_assets() -> None:
    config_path = Path("configs/static-phase1.yaml").resolve()

    manifest = verify_environment(config_path, require_cuda=False)

    assert manifest.ultralytics == "8.4.90"
    assert manifest.faster_coco_eval == "1.7.2"
    assert manifest.onnx == "1.22.0"
    assert manifest.ml_dtypes == "0.5.4"
    assert manifest.protobuf == "7.35.1"
    assert manifest.source_weights_hash == "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d"
    assert manifest.official_model_yaml_hash == "43d8a7c86acc77282ddf4966d6526e091e5b064c140ed4a6e4c0b68ecbc3784c"
    assert manifest.official_default_yaml_hash == "f6e19ab7228826341cc6370bcc8c78d10478d2bd604e25a13d9975b5efc9b410"


def test_pipeline_identity_changes_with_any_research_input_hash() -> None:
    first = pipeline_identity("config-a", "data-a", "environment-a")

    assert first == pipeline_identity("config-a", "data-a", "environment-a")
    assert first != pipeline_identity("config-b", "data-a", "environment-a")
    assert first != pipeline_identity("config-a", "data-b", "environment-a")
    assert len(first) == 12


def test_gpu_scheduler_requires_three_consecutive_idle_observations() -> None:
    snapshots = iter(
        [
            GpuSnapshot(25000, 5),
            GpuSnapshot(12000, 80),
            GpuSnapshot(26000, 3),
            GpuSnapshot(27000, 2),
            GpuSnapshot(28000, 1),
        ]
    )
    sleeps: list[float] = []

    result = wait_for_gpu_idle(
        query=lambda: next(snapshots),
        sleep=sleeps.append,
        poll_interval_seconds=7,
    )

    assert result["ready"] is True
    assert len(sleeps) == 4
    assert [item["free_memory_mib"] for item in result["observations"]] == [26000, 27000, 28000]


def test_scheduler_waits_for_original_gpu_service_before_gpu_idle_gate() -> None:
    states = iter([True, True, False])
    sleeps: list[float] = []

    result = wait_for_predecessor_units(
        ("yolo-p2-study.service",),
        query=lambda _unit: next(states),
        sleep=sleeps.append,
        poll_interval_seconds=11,
    )

    assert result["ready"] is True
    assert result["units"] == ["yolo-p2-study.service"]
    assert sleeps == [11.0, 11.0]
