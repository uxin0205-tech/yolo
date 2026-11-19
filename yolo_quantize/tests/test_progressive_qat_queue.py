from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest

from yolo_quantize.progressive_qat_queue import ProgressiveShortQATMaterializer
from yolo_quantize.progressive_queue import ProgressiveQueuePlan
from yolo_quantize.qat_plan import Full35QATPlan
from yolo_quantize.weight_quantization import FixedSD4WeightSpec, UniformWeightSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PLAN = PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _short_queue(tmp_path: Path) -> Path:
    plan = ProgressiveQueuePlan.from_yaml(QUEUE_PLAN)
    candidate_id = "backbone-early-primary--fixed_sd4"
    record = {
        "candidate_id": candidate_id,
        "format_id": "fixed_sd4",
        "weight_format_id": "fixed-sd4",
        "cpu_format_id": ("fixed-sd4-per_output_channel-optimal_scaled_codebook"),
        "assignments": [
            {
                "stage_id": "backbone-early-primary",
                "region": "backbone_early",
                "path": "graph.model.4.m.0.cv3.conv",
                "format_id": "fixed_sd4",
                "weight_format_id": "fixed-sd4",
                "cpu_format_id": (
                    "fixed-sd4-per_output_channel-optimal_scaled_codebook"
                ),
            }
        ],
        "metrics": dict(plan.locked_parent_metrics),
        "metrics_path": str(plan.locked_parent_metrics_path),
        "metrics_sha256": _sha256(plan.locked_parent_metrics_path),
        "packed_bytes": 22600000,
        "savings_from_locked_parent_bytes": 87936,
        "gate": {
            "decision": "recover",
            "worst_total_map50_delta": -0.02,
            "worst_total_map50_95_delta": -0.03,
        },
        "seconds": 1.0,
        "status": "completed",
    }
    path = tmp_path / "short-qat-queue.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queue_id": f"{plan.queue_id}--short-qat",
                "source_queue_plan": str(plan.config_path),
                "source_queue_plan_sha256": plan.config_sha256,
                "source_parent_manifest": str(plan.parent.config_path),
                "source_parent_manifest_sha256": plan.parent.config_sha256,
                "status": "ready_for_qat_plan_materialization",
                "execution_authorized": True,
                "matched_sham_required": True,
                "recipe": {
                    "optimizer": "AdamW",
                    "epochs": 15,
                    "patience": 5,
                    "schedule": {
                        "fp32_epochs": 3,
                        "progressive_ramp_epochs": 6,
                        "full_quant_epochs": 6,
                    },
                    "batch": {
                        "detect_logical": 128,
                        "detect_microbatch": 16,
                        "pose": 16,
                    },
                    "added_noise": False,
                    "augmentation": "accepted_full35_exact",
                },
                "jobs": [record],
                "formal_validation": False,
                "long_qat_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_materializer_builds_hash_pinned_v19_warm_started_matched_qat_plan(
    tmp_path: Path,
) -> None:
    source = _short_queue(tmp_path)
    output = tmp_path / "materialized"
    materializer = ProgressiveShortQATMaterializer(
        source,
        output_root=output,
    )

    manifest = materializer.materialize()
    repeated = materializer.materialize()

    assert manifest == repeated
    assert manifest["status"] == "ready"
    assert len(manifest["jobs"]) == 1
    plan_path = Path(manifest["jobs"][0]["plan"])
    plan = Full35QATPlan.from_yaml(plan_path)
    assert plan.warm_start is not None
    assert plan.warm_start.parent_id == "v19-poly-shift-a8-all-w8-qat-epoch5"
    assert plan.warm_start.load_optimizer_state is False
    assert (
        plan.matched_metrics
        == ProgressiveQueuePlan.from_yaml(QUEUE_PLAN).locked_parent_metrics_path
    )
    assert plan.training.epochs == 15
    assert plan.training.patience == 5
    assert plan.training.progressive_start_epoch == 2
    assert plan.training.progressive_full_epoch == 9
    assert plan.training.scale_only_epochs == 9
    assert len(plan.assignments) == 11
    assert all(
        isinstance(assignment.spec, UniformWeightSpec)
        for assignment in plan.assignments[:10]
    )
    assert isinstance(plan.assignments[-1].spec, FixedSD4WeightSpec)
    assert plan.assignments[-1].paths == ("graph.model.4.m.0.cv3.conv",)
    assert Path(manifest["jobs"][0]["evidence"]).is_file()
    assert Path(manifest["jobs"][0]["dual_regate"]).is_file()


def test_materializer_emits_complete_no_jobs_manifest(tmp_path: Path) -> None:
    source = _short_queue(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["status"] = "no_eligible_recovery_candidates"
    payload["jobs"] = []
    source.write_text(json.dumps(payload), encoding="utf-8")

    manifest = ProgressiveShortQATMaterializer(
        source,
        output_root=tmp_path / "empty",
    ).materialize()

    assert manifest["status"] == "complete_no_jobs"
    assert manifest["jobs"] == []


class _FakeRuntime:
    calls: ClassVar[list[tuple[str, str, str | None]]] = []

    def __init__(self, plan_path: Path) -> None:
        self.plan = Full35QATPlan.from_yaml(plan_path)

    @classmethod
    def from_yaml(cls, plan_path: Path):
        return cls(plan_path)

    def run_name(self, arm: str) -> str:
        return f"{self.plan.plan_id}-{arm}-seed{self.plan.training.seed}"

    def run(self, arm: str, *, device_index: int, resume=None):
        self.calls.append(
            (
                self.plan.candidate_id,
                arm,
                None if resume is None else str(resume),
            )
        )
        run_dir = self.plan.run_root / self.run_name(arm)
        checkpoint = run_dir / "checkpoints" / "best_joint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{self.plan.candidate_id}:{arm}".encode())
        (run_dir / "qat-experiment.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "plan_sha256": self.plan.config_sha256,
                    "candidate_id": self.plan.candidate_id,
                    "arm": arm,
                    "run_name": self.run_name(arm),
                    "completed_stages": ["j3"],
                    "epochs_completed": self.plan.training.epochs,
                    "formal_validation": False,
                }
            ),
            encoding="utf-8",
        )
        return object()


def test_short_qat_runner_executes_sham_then_qat_once_and_resumes_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from yolo_quantize.progressive_qat_queue import ProgressiveShortQATRunner

    source = _short_queue(tmp_path)
    output = tmp_path / "runner"
    _FakeRuntime.calls = []
    monkeypatch.setattr(
        "yolo_quantize.progressive_qat_queue._PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr("yolo_quantize.qat_plan.PROJECT_ROOT", tmp_path)
    runner = ProgressiveShortQATRunner(
        source,
        output_root=output,
        runtime_factory=_FakeRuntime.from_yaml,
        gpu_pid_probe=lambda device: (),
        sleep=lambda seconds: None,
    )

    completed = runner.run()
    resumed = runner.run()

    assert completed["status"] == "complete"
    assert resumed == completed
    assert [arm for _, arm, _ in _FakeRuntime.calls] == ["sham", "qat"]
    assert all(resume is None for _, _, resume in _FakeRuntime.calls)
    state = json.loads((output / "execution-state.json").read_text())
    assert state["jobs"][0]["arms"] == {
        "sham": {"status": "completed"},
        "qat": {"status": "completed"},
    }


class _NonRetryableError(RuntimeError):
    retryable = False


class _NonRetryableRuntime(_FakeRuntime):
    def run(self, arm: str, *, device_index: int, resume=None):
        if arm == "qat":
            self.calls.append((self.plan.candidate_id, arm, None))
            raise _NonRetryableError("matched sham evidence is invalid")
        return super().run(arm, device_index=device_index, resume=resume)


def test_short_qat_runner_does_not_retry_non_retryable_precondition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from yolo_quantize.progressive_qat_queue import ProgressiveShortQATRunner

    source = _short_queue(tmp_path)
    _NonRetryableRuntime.calls = []
    monkeypatch.setattr(
        "yolo_quantize.progressive_qat_queue._PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr("yolo_quantize.qat_plan.PROJECT_ROOT", tmp_path)
    runner = ProgressiveShortQATRunner(
        source,
        output_root=tmp_path / "runner-non-retryable",
        runtime_factory=_NonRetryableRuntime.from_yaml,
        gpu_pid_probe=lambda device: (),
        sleep=lambda seconds: None,
    )

    with pytest.raises(_NonRetryableError):
        runner.run()

    assert [arm for _, arm, _ in _NonRetryableRuntime.calls] == ["sham", "qat"]


def test_materializer_supports_distinct_qsilu_namespace_and_run_root(
    tmp_path: Path,
) -> None:
    source = _short_queue(tmp_path)
    qat_root = PROJECT_ROOT / "artifacts/runs/qat/v31-qsilu-progressive-short"
    manifest = ProgressiveShortQATMaterializer(
        source,
        output_root=tmp_path / "materialized-custom",
        plan_namespace="v31-qsilu",
        qat_run_root=qat_root,
        authorization_id="user-2026-09-05-complete-qsilu-quantization-lane",
        plan_date="2026-09-05",
    ).materialize()

    plan = Full35QATPlan.from_yaml(Path(manifest["jobs"][0]["plan"]))
    assert plan.plan_id.startswith("v31-qsilu-short-qat-")
    assert plan.run_root == qat_root.resolve()
    assert (
        plan.execution_authorization_id
        == "user-2026-09-05-complete-qsilu-quantization-lane"
    )
