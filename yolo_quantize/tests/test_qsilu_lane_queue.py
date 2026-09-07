from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_quantize.qsilu_lane_queue import (
    DependencyFailedError,
    QueueRecorder,
    wait_for_dependency,
)


def _write_state(path: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status}), encoding="utf-8")


def test_dependency_wait_uses_600_second_event_only_poll(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency.json"
    events = tmp_path / "queue"
    _write_state(dependency, "running")
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        _write_state(dependency, "complete")

    payload = wait_for_dependency(
        dependency,
        required_status="complete",
        poll_seconds=600,
        recorder=QueueRecorder(events, "fixture"),
        sleep=sleep,
    )

    assert payload["status"] == "complete"
    assert sleeps == [600.0]
    records = [
        json.loads(line) for line in (events / "events.jsonl").read_text().splitlines()
    ]
    assert [record["kind"] for record in records] == [
        "waiting_for_dependency",
        "dependency_complete",
    ]
    assert records[0]["values"]["poll_seconds"] == 600


def test_dependency_failure_stops_without_polling(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency.json"
    _write_state(dependency, "failed")

    with pytest.raises(DependencyFailedError, match="failed"):
        wait_for_dependency(
            dependency,
            required_status="complete",
            poll_seconds=600,
            recorder=QueueRecorder(tmp_path / "queue", "fixture"),
            sleep=lambda seconds: pytest.fail(
                "must not sleep after dependency failure"
            ),
        )


def test_dependency_hash_drift_stops_without_polling(tmp_path: Path) -> None:
    import hashlib

    dependency = tmp_path / "dependency.json"
    _write_state(dependency, "archived_for_qsilu_handoff")
    expected = hashlib.sha256(dependency.read_bytes()).hexdigest()
    _write_state(dependency, "drifted")

    with pytest.raises(DependencyFailedError, match="hash mismatch"):
        wait_for_dependency(
            dependency,
            required_status="archived_for_qsilu_handoff",
            expected_sha256=expected,
            poll_seconds=600,
            recorder=QueueRecorder(tmp_path / "queue", "fixture"),
            sleep=lambda seconds: pytest.fail("must not sleep after hash drift"),
        )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANE_PLAN = (
    PROJECT_ROOT / "configs/experiments/v30-qsilu-complete-quantization-lane-v1.yaml"
)


def test_lane_plan_pins_qsilu_all_ten_w8_and_600_seconds() -> None:
    from yolo_quantize.qsilu_lane_queue import QSiluLanePlan

    plan = QSiluLanePlan.from_yaml(LANE_PLAN)

    assert plan.poll_seconds == 600
    assert plan.required_dependency_status == "archived_for_qsilu_handoff"
    assert plan.dependency_sha256 == (
        "ce73d390f2c62421d7c98a26e5b46e980d89bc5443641bb7956e87031cdffb17"
    )
    assert plan.q1_candidate_id == "qsilu-all-ten-regions-w8"
    assert len(plan.q1.candidates[0].region_defaults) == 10
    assert plan.q1.activation_policy.activation == "qsilu_pq"
    assert plan.q2_plan_output.name == "v30-qsilu-all-w8-qat-pilot-v1.yaml"
    assert plan.q3_parent_manifest.name == "v30-qsilu-all-w8-locked-parent-v1.yaml"
    assert plan.q4_profile_id == "v30-qsilu-all-w8-progressive-weight-formats-cpu-v1"
    assert plan.q5_run_root.name == "v31-qsilu-progressive-ptq-to-short-qat-v1"
    assert plan.q6_qat_run_root.name == "v31-qsilu-progressive-short"


def test_completed_q1_report_must_pin_plan_and_candidate(tmp_path: Path) -> None:
    from yolo_quantize.qsilu_lane_queue import QSiluLanePlan, validate_q1_report

    plan = QSiluLanePlan.from_yaml(LANE_PLAN)
    report = tmp_path / "q1.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "kind": "full35_mixed_weight_policy_map50_search",
                "formal_training": False,
                "formal_validation": False,
                "contract": {
                    "reviewed_plan": {
                        "plan_id": plan.q1.plan_id,
                        "plan_sha256": plan.q1.config_sha256,
                    }
                },
                "results": {
                    plan.q1_candidate_id: {
                        "status": "completed",
                        "gate": {"decision": "recover"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_q1_report(plan, report)
    assert result["decision"] == "recover"

    payload = json.loads(report.read_text())
    payload["contract"]["reviewed_plan"]["plan_sha256"] = "drifted"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="plan hash"):
        validate_q1_report(plan, report)


def test_q2_materializer_builds_qsilu_owned_all_w8_paired_qat(tmp_path: Path) -> None:
    from yolo_quantize.qat_plan import Full35QATPlan
    from yolo_quantize.qsilu_lane_queue import QSiluLanePlan, materialize_q2_plan

    lane = QSiluLanePlan.from_yaml(LANE_PLAN)
    q1 = tmp_path / "q1.json"
    q1.write_text(
        json.dumps(
            {
                "status": "completed",
                "results": {
                    lane.q1_candidate_id: {
                        "status": "completed",
                        "gate": {"decision": "recover"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    dual = tmp_path / "dual.json"
    dual.write_text(
        json.dumps(
            {
                "status": "completed",
                "source": {
                    "report_sha256": hashlib.sha256(q1.read_bytes()).hexdigest()
                },
                "candidates": {lane.q1_candidate_id: {"decision": "recover"}},
            }
        ),
        encoding="utf-8",
    )

    output = materialize_q2_plan(
        lane, q1_report=q1, dual_report=dual, output_path=tmp_path / "q2.yaml"
    )
    plan = Full35QATPlan.from_yaml(output)

    assert plan.activation.activation == "qsilu_pq"
    assert (
        plan.parent_checkpoint_sha256
        == "7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e"
    )
    assert plan.candidate_id == "qsilu-all-ten-regions-w8"
    assert len(plan.assignments) == 10
    assert {assignment.spec.format_id for assignment in plan.assignments} == {"w8"}
    assert plan.training.epochs == 15
    assert plan.training.patience == 5
    assert plan.training.detect_logical_batch == 128
    assert plan.training.detect_microbatch == 16
    assert plan.warm_start is None


def test_paired_qat_executes_sham_then_qat_with_one_runtime_contract(
    tmp_path: Path,
) -> None:
    import hashlib

    from yolo_quantize.qat_plan import Full35QATPlan
    from yolo_quantize.qsilu_lane_queue import (
        QSiluLanePlan,
        materialize_q2_plan,
        run_paired_qat,
    )

    lane = QSiluLanePlan.from_yaml(LANE_PLAN)
    q1 = tmp_path / "q1.json"
    q1.write_text(
        json.dumps(
            {
                "status": "completed",
                "results": {
                    lane.q1_candidate_id: {
                        "status": "completed",
                        "gate": {"decision": "recover"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    dual = tmp_path / "dual.json"
    dual.write_text(
        json.dumps(
            {
                "status": "completed",
                "source": {
                    "report_sha256": hashlib.sha256(q1.read_bytes()).hexdigest()
                },
                "candidates": {lane.q1_candidate_id: {"decision": "recover"}},
            }
        ),
        encoding="utf-8",
    )
    plan_path = materialize_q2_plan(
        lane, q1_report=q1, dual_report=dual, output_path=tmp_path / "q2.yaml"
    )

    calls: list[tuple[str, str | None]] = []

    class FakeRuntime:
        def __init__(self, path: Path) -> None:
            from dataclasses import replace

            self.plan = replace(
                Full35QATPlan.from_yaml(path), run_root=tmp_path / "runs"
            )

        def run_name(self, arm: str) -> str:
            return f"{self.plan.plan_id}-{arm}-seed{self.plan.training.seed}"

        def run(self, arm: str, *, device_index: int, resume=None):
            calls.append((arm, None if resume is None else str(resume)))
            run = self.plan.run_root / self.run_name(arm)
            run.mkdir(parents=True, exist_ok=True)
            (run / "qat-experiment.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "plan_sha256": self.plan.config_sha256,
                        "candidate_id": self.plan.candidate_id,
                        "arm": arm,
                        "run_name": self.run_name(arm),
                        "completed_stages": ["j3"],
                        "formal_validation": False,
                    }
                ),
                encoding="utf-8",
            )

    result = run_paired_qat(
        plan_path,
        device_index=0,
        maximum_retries_per_arm=1,
        recorder=QueueRecorder(tmp_path / "events", "fixture"),
        runtime_factory=FakeRuntime,
        gpu_pid_probe=lambda device: (),
        sleep=lambda seconds: None,
    )

    assert calls == [("sham", None), ("qat", None)]
    assert result["arms"] == {"sham": "completed", "qat": "completed"}


def test_q2_preflight_uses_resolved_plan_hash_and_reuses_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from yolo_quantize import qat_runtime
    from yolo_quantize.qsilu_lane_queue import ensure_q2_preflight

    plan_sha256 = "a" * 64
    report = {
        "ready": True,
        "blockers": [],
        "resolved": {"plan_sha256": plan_sha256},
    }
    runtime = SimpleNamespace(
        plan=SimpleNamespace(config_sha256=plan_sha256),
        preflight=lambda **kwargs: SimpleNamespace(to_dict=lambda: report),
    )
    monkeypatch.setattr(
        qat_runtime.Full35QATRuntime,
        "from_yaml",
        staticmethod(lambda path: runtime),
    )
    output = tmp_path / "preflight.json"

    assert ensure_q2_preflight(tmp_path / "plan.yaml", output) == report
    assert ensure_q2_preflight(tmp_path / "plan.yaml", output) == report


def test_best_joint_selector_and_dual_gate_replay_v19() -> None:
    from yolo_quantize.qsilu_lane_queue import (
        evaluate_total_dual_gate,
        select_best_joint_epoch,
    )

    run = (
        PROJECT_ROOT / "artifacts/runs/qat/v19-poly-shift-all-w8-qat-pilot-v1-qat-seed1"
    )
    epoch, score = select_best_joint_epoch(run / "logs/gate.csv")
    gate = evaluate_total_dual_gate(
        PROJECT_ROOT
        / "artifacts/runs/v15-poly-shift-backbone-early-w8-search-v1/validation/accepted/epoch-0000/bittrue/metrics.json",
        run / f"validation/epoch-{epoch:04d}/bittrue/metrics.json",
    )

    assert epoch == 5
    assert score == pytest.approx(0.8317651484021804)
    assert gate["decision"] == "green"
    assert gate["worst_map50_delta"] == pytest.approx(-0.014133354771334816)
    assert gate["worst_map50_95_delta"] == pytest.approx(-0.012950485942226253)
    assert len(gate["total_deltas"]) == 16


def test_locked_parent_builder_replays_existing_v19_without_selector_drift(
    tmp_path: Path,
) -> None:
    from yolo_quantize.progressive_preparation import LockedQATParentSpec
    from yolo_quantize.qsilu_lane_queue import lock_qat_parent

    output = lock_qat_parent(
        PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml",
        output_path=tmp_path / "locked.yaml",
        parent_id="fixture-v19-epoch5",
    )
    parent = LockedQATParentSpec.from_yaml(output)

    assert parent.selected_epoch == 5
    assert parent.activation_name == "poly_shift"
    assert parent.deployment_modules == 148
    assert parent.worst_map50_delta == pytest.approx(-0.014133354771334816)
    assert parent.worst_map50_95_delta == pytest.approx(-0.012950485942226253)


def test_cpu_profile_accepts_qsilu_specific_profile_identity() -> None:
    import inspect

    from yolo_quantize.progressive_preparation import ProgressiveWeightPreparation

    signature = inspect.signature(ProgressiveWeightPreparation.prepare_profile)
    assert "profile_id" in signature.parameters


def test_progressive_plan_materializer_replays_v29_schema_at_600_seconds(
    tmp_path: Path,
) -> None:
    from yolo_quantize.progressive_queue import ProgressiveQueuePlan
    from yolo_quantize.qsilu_lane_queue import materialize_progressive_plan

    output = materialize_progressive_plan(
        locked_parent=PROJECT_ROOT
        / "artifacts/manifests/v19-epoch5-locked-parent-v1.yaml",
        cpu_profile=PROJECT_ROOT
        / "artifacts/reports/v19-epoch5-progressive-weight-formats-cpu-v1.json",
        output_path=tmp_path / "progressive.yaml",
        queue_id="fixture-progressive",
        run_root=PROJECT_ROOT / "artifacts/queues/fixture-progressive",
    )
    plan = ProgressiveQueuePlan.from_yaml(output)

    assert plan.poll_seconds == 600
    assert plan.parent.parent_id == "v19-poly-shift-a8-all-w8-qat-epoch5"
    assert len(plan.stages) == 10
    assert plan.short_qat.patience == 5
    assert plan.recover_map50_floor == pytest.approx(0.04)
    assert plan.recover_map50_95_floor == pytest.approx(0.08)


def test_qsilu_progressive_plan_adds_intermediate_bits_to_every_stage(
    tmp_path: Path,
) -> None:
    import copy

    from yolo_quantize.progressive_queue import ProgressiveQueuePlan
    from yolo_quantize.qsilu_lane_queue import materialize_progressive_plan

    source_profile = (
        PROJECT_ROOT
        / "artifacts/reports/v19-epoch5-progressive-weight-formats-cpu-v1.json"
    )
    profile = json.loads(source_profile.read_text(encoding="utf-8"))
    for row in profile["summary"]["path_rankings"]:
        base = row["formats"]["uniform-w4-per_output_channel-optimal_scaled_codebook"]
        for bits in (7, 6, 5):
            encoded = copy.deepcopy(base)
            encoded["bits"] = bits
            encoded["code_bytes"] = (int(row["elements"]) * bits + 7) // 8
            encoded["packed_bytes"] = encoded["code_bytes"] + encoded["metadata_bytes"]
            row["formats"][
                f"uniform-w{bits}-per_output_channel-optimal_scaled_codebook"
            ] = encoded
    profile["summary"]["coverage"]["measurements"] = 148 * 8
    profile["summary"]["coverage"]["formats_per_path"] = 8
    cpu_profile = tmp_path / "profile.json"
    cpu_profile.write_text(json.dumps(profile), encoding="utf-8")

    output = materialize_progressive_plan(
        locked_parent=(
            PROJECT_ROOT / "artifacts/manifests/v19-epoch5-locked-parent-v1.yaml"
        ),
        cpu_profile=cpu_profile,
        output_path=tmp_path / "progressive-8.yaml",
        queue_id="fixture-progressive-8",
        run_root=PROJECT_ROOT / "artifacts/queues/fixture-progressive-8",
        include_intermediate_uniform_bits=True,
    )
    plan = ProgressiveQueuePlan.from_yaml(output)

    expected = {
        "exact_w7",
        "exact_w6",
        "exact_w5",
        "exact_w4",
        "fixed_sd4",
        "paper_twn_v2",
        "twn_v3_filterwise",
        "exact_scaled_ternary",
    }
    assert set(plan.format_catalog) == expected
    assert all(
        {item.format_id for item in stage.formats} == expected for stage in plan.stages
    )


def test_intermediate_uniform_ptq_candidates_round_trip_to_qat() -> None:
    from yolo_quantize.mixed_policy_search import parse_weight_format_spec
    from yolo_quantize.progressive_qat_queue import _format_payload

    for bits in (7, 6, 5, 4):
        payload = _format_payload(f"exact_w{bits}")
        spec = parse_weight_format_spec(payload)
        assert spec.bits == bits
        assert spec.format_id == f"w{bits}"


def test_full_lane_orchestrates_q1_through_q7_without_path_collisions(
    tmp_path: Path, monkeypatch
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    import yolo_quantize.qsilu_lane_queue as module
    from yolo_quantize.qsilu_lane_queue import QSiluLanePlan, QSiluLaneQueue

    dependency = tmp_path / "dependency.json"
    _write_state(dependency, "archived_for_qsilu_handoff")
    lane = replace(
        QSiluLanePlan.from_yaml(LANE_PLAN),
        dependency_state=dependency,
        dependency_sha256=__import__("hashlib")
        .sha256(dependency.read_bytes())
        .hexdigest(),
        q1_output=tmp_path / "q1.json",
        q1_dual_output=tmp_path / "q1-dual.json",
        q2_plan_output=tmp_path / "q2.yaml",
        q2_preflight=tmp_path / "q2-preflight.json",
        q3_parent_manifest=tmp_path / "parent.yaml",
        q4_profile=tmp_path / "profile.json",
        q5_plan_output=tmp_path / "q5.yaml",
        q5_run_root=tmp_path / "q5-run",
        q6_output_root=tmp_path / "q6",
        q6_qat_run_root=tmp_path / "q6-runs",
        q7_report=tmp_path / "summary.json",
    )
    lane.q1_output.write_text("{}", encoding="utf-8")
    q2 = lane.q2_plan_output
    parent = lane.q3_parent_manifest
    profile = lane.q4_profile
    q5 = lane.q5_plan_output
    for path in (q2, parent, profile, q5):
        path.write_text(path.name, encoding="utf-8")
    queue = QSiluLaneQueue(
        lane,
        queue_root=tmp_path / "queue",
        gpu_pid_probe=lambda device: (),
        sleep=lambda seconds: None,
    )
    monkeypatch.setattr(
        queue, "_run_q1", lambda: {"report_sha256": module._sha256(lane.q1_output)}
    )
    monkeypatch.setattr(
        module,
        "regate_candidate_report",
        lambda **kwargs: {
            "status": "completed",
            "candidates": {lane.q1_candidate_id: {"decision": "recover"}},
        },
    )
    monkeypatch.setattr(module, "materialize_q2_plan", lambda *args, **kwargs: q2)
    monkeypatch.setattr(
        module, "ensure_q2_preflight", lambda *args, **kwargs: {"ready": True}
    )
    monkeypatch.setattr(
        module,
        "run_paired_qat",
        lambda *args, **kwargs: {
            "plan": str(q2),
            "plan_sha256": "q2",
            "arms": {"sham": "completed", "qat": "completed"},
        },
    )
    monkeypatch.setattr(module, "lock_qat_parent", lambda *args, **kwargs: parent)
    monkeypatch.setattr(
        module,
        "ensure_cpu_profile",
        lambda *args, **kwargs: {"summary": {"coverage": {"measurements": 740}}},
    )
    monkeypatch.setattr(module, "materialize_progressive_plan", lambda **kwargs: q5)
    import yolo_quantize.progressive_queue as progressive

    monkeypatch.setattr(
        progressive.ProgressiveQueuePlan, "from_yaml", lambda path: SimpleNamespace()
    )
    monkeypatch.setattr(
        progressive,
        "ProgressiveExperimentQueue",
        lambda plan: SimpleNamespace(
            run=lambda: {"status": "ptq_complete_short_qat_ready"}
        ),
    )
    lane.q5_run_root.mkdir(parents=True)
    (lane.q5_run_root / "short-qat-queue.json").write_text("{}")
    import yolo_quantize.progressive_qat_queue as short

    monkeypatch.setattr(
        short,
        "ProgressiveShortQATRunner",
        lambda *args, **kwargs: SimpleNamespace(
            run=lambda: {"status": "complete", "jobs": []}
        ),
    )
    monkeypatch.setattr(
        module,
        "_complete_lane_summary",
        lambda *args, **kwargs: {
            "schema_version": 1,
            "status": "complete_search_pipeline_pending_finalist_review",
        },
    )

    result = queue.run()
    assert result["status"] == "complete_search_pipeline_pending_finalist_review"
    assert lane.q7_report.is_file()
