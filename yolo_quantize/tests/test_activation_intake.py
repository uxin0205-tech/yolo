from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yolo_quantize import ActivationIntake
from yolo_quantize.intake import main


def _write_pending_queue(root: Path) -> None:
    queue_config = root / "training/full35/experiment-queue.yaml"
    queue_config.parent.mkdir(parents=True)
    queue_config.write_text(
        """\
queue_id: full35-activation-seed1-v1
execution:
  state: artifacts/runs/full35/queue/queue-state.json
""",
        encoding="utf-8",
    )
    state = root / "artifacts/runs/full35/queue/queue-state.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "queue": "experiment-queue",
                "next_job": "finalist-seed1-qsilu-pq",
                "counts": {"completed": 1, "pending": 1, "blocked": 0},
                "jobs": [],
            }
        ),
        encoding="utf-8",
    )


def test_intake_refuses_to_be_ready_while_activation_queue_is_pending(
    tmp_path: Path,
) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    full35_root.mkdir()
    _write_pending_queue(activation_root)

    report = ActivationIntake.from_workspace(
        activation_root=activation_root,
        full35_root=full35_root,
    ).analyze()

    assert not report.ready
    assert "experiment-queue has 1 pending job(s)" in report.blockers
    assert report.queues[0].queue_name == "experiment-queue"
    assert report.queues[0].pending == 1
    assert report.queues[0].next_job == "finalist-seed1-qsilu-pq"
    assert not report.queues[0].terminal
    assert not report.queues[0].runner_active
    assert (
        "experiment-queue is interrupted: pending jobs remain but its runner "
        "lock is not held"
    ) in report.blockers


def test_intake_reports_missing_queue_state_as_a_blocker(tmp_path: Path) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    full35_root.mkdir()
    queue_config = activation_root / "training/full35/experiment-queue.yaml"
    queue_config.parent.mkdir(parents=True)
    queue_config.write_text(
        """\
queue_id: full35-activation-seed1-v1
execution:
  state: artifacts/runs/full35/queue/missing-state.json
""",
        encoding="utf-8",
    )

    report = ActivationIntake.from_workspace(
        activation_root=activation_root,
        full35_root=full35_root,
    ).analyze()

    assert not report.ready
    assert (
        "experiment-queue state is missing: "
        "artifacts/runs/full35/queue/missing-state.json"
    ) in report.blockers


def test_intake_fails_closed_when_no_queue_definitions_exist(tmp_path: Path) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    activation_root.mkdir()
    full35_root.mkdir()

    report = ActivationIntake.from_workspace(
        activation_root=activation_root,
        full35_root=full35_root,
    ).analyze()

    assert not report.ready
    assert "no activation queue definitions were found" in report.blockers


def test_intake_report_has_a_stable_json_shape(tmp_path: Path) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    full35_root.mkdir()
    _write_pending_queue(activation_root)

    report = ActivationIntake.from_workspace(
        activation_root=activation_root,
        full35_root=full35_root,
    ).analyze()
    payload = report.to_dict()

    assert payload["schema_version"] == 2
    assert payload["ready"] is False
    assert payload["queues"][0] == {
        "queue_name": "experiment-queue",
        "queue_id": "full35-activation-seed1-v1",
        "config_path": "training/full35/experiment-queue.yaml",
        "state_path": "artifacts/runs/full35/queue/queue-state.json",
        "completed": 1,
        "pending": 1,
        "blocked": 0,
        "next_job": "finalist-seed1-qsilu-pq",
        "updated_at": None,
        "runner_active": False,
        "terminal": False,
    }
    json.dumps(payload)


def test_intake_cli_prints_json_and_returns_blocked_exit_code(
    tmp_path: Path,
    capsys,
) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    full35_root.mkdir()
    _write_pending_queue(activation_root)

    exit_code = main(
        [
            "--activation-root",
            str(activation_root),
            "--full35-root",
            str(full35_root),
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["queues"][0]["pending"] == 1


def test_upstream_gate_failure_within_quantization_tolerance_remains_candidate(
    tmp_path: Path,
) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    full35_root.mkdir()
    queue_config = activation_root / "training/full35/experiment-queue.yaml"
    baseline = activation_root / "training/full35/contracts/baseline.yaml"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        """\
metrics:
  coco/box/map50_95: 0.50
  bbat/box/map50_95: 0.63
  bbat/pose/map50_95: 0.90
""",
        encoding="utf-8",
    )
    queue_config.write_text(
        """\
queue_id: full35-activation-seed1-v1
execution:
  state: artifacts/runs/full35/queue/queue-state.json
accuracy_gate:
  baseline: contracts/baseline.yaml
jobs:
  - id: short-recovery-uniform-poly-quality
    kind: train
    activation: poly_quality
    phase: short_recovery
    run_name: poly-quality-run
""",
        encoding="utf-8",
    )
    state_path = activation_root / "artifacts/runs/full35/queue/queue-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "queue": "experiment-queue",
                "next_job": None,
                "counts": {"completed": 1, "pending": 0, "blocked": 0},
                "jobs": [
                    {
                        "id": "short-recovery-uniform-poly-quality",
                        "run": "poly-quality-run",
                        "status": "completed",
                        "gate": {
                            "passed": False,
                            "worst_delta": -0.02,
                            "failed_metrics": ["coco/box/map50_95"],
                            "deltas": {
                                "coco/box/map50_95": -0.02,
                                "bbat/box/map50_95": -0.01,
                                "bbat/pose/map50_95": -0.003,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = ActivationIntake.from_workspace(
        activation_root=activation_root,
        full35_root=full35_root,
    ).analyze()

    job = report.jobs[0]
    assert job.activation == "poly_quality"
    assert job.upstream_gate_status == "failed"
    assert job.quantization_eligibility == "screen_candidate"
    assert job.gate_passed is False
    assert job.metrics == {
        "coco/box/map50_95": 0.48,
        "bbat/box/map50_95": 0.62,
        "bbat/pose/map50_95": 0.897,
    }
    assert job.primary_metrics == {
        "coco/box/map50_95": 0.48,
        "bbat/box/map50_95": 0.62,
        "bbat/pose/map50_95": 0.897,
    }
    assert (
        "short-recovery-uniform-poly-quality completed train run is missing: "
        "artifacts/runs/full35/poly-quality-run/activation-experiment.json"
    ) in report.blockers


def test_intake_hashes_critical_qsilu_source_with_provenance(tmp_path: Path) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    full35_root.mkdir()
    _write_pending_queue(activation_root)
    source = activation_root / "src/activation_lab/activations.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reviewed qsilu source\n")

    report = ActivationIntake.from_workspace(
        activation_root=activation_root,
        full35_root=full35_root,
    ).analyze()

    digest = next(item for item in report.files if item.role == "qsilu_float_source")
    assert digest.owner == "yolo_activation"
    assert digest.relative_path == "src/activation_lab/activations.py"
    assert digest.bytes == len(b"reviewed qsilu source\n")
    assert digest.sha256 == hashlib.sha256(b"reviewed qsilu source\n").hexdigest()


def test_full35_manifest_does_not_require_a_self_referential_checksum(
    tmp_path: Path,
) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    activation_root.mkdir()
    payloads = {
        "MANIFEST.json": b"{}\n",
        "RELEASE_STATUS.json": b"{}\n",
        "configs/joint.yaml": b"schema_version: 1\n",
        "weights/combined/inference/best_joint.pt": b"checkpoint",
    }
    checksum_lines = []
    for relative_path, content in payloads.items():
        path = full35_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if relative_path != "MANIFEST.json":
            checksum_lines.append(
                f"{hashlib.sha256(content).hexdigest()}  {relative_path}"
            )
    (full35_root / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    report = ActivationIntake.from_workspace(
        activation_root=activation_root,
        full35_root=full35_root,
    ).analyze()

    assert (
        "Full35 checksum is missing for required file: MANIFEST.json"
        not in report.blockers
    )


def test_intake_reports_malformed_queue_state_instead_of_raising(
    tmp_path: Path,
) -> None:
    activation_root = tmp_path / "yolo_activation"
    full35_root = tmp_path / "full35"
    full35_root.mkdir()
    _write_pending_queue(activation_root)
    state_path = activation_root / "artifacts/runs/full35/queue/queue-state.json"
    state_path.write_text("{not-json", encoding="utf-8")

    report = ActivationIntake.from_workspace(
        activation_root=activation_root,
        full35_root=full35_root,
    ).analyze()

    assert not report.ready
    assert "experiment-queue state is malformed JSON" in report.blockers
