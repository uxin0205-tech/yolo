from __future__ import annotations

from pathlib import Path
import json

import pytest

from masf_yolo.contracts import PipelineState
from masf_yolo.workflow import PHASE1_STAGES, PipelineWorkflow, StageResult


def test_phase1_dag_has_selection_before_test_and_no_phase2_nodes() -> None:
    names = [stage.name for stage in PHASE1_STAGES]

    assert names.index("selection") < names.index("test_all")
    assert names[-2:] == ["final_audit", "report"]
    assert not any("s0" in name.lower() or "temporal" in name.lower() for name in names)
    for index, stage in enumerate(PHASE1_STAGES):
        assert set(stage.dependencies) <= set(names[:index])


def test_m7_is_gated_and_completed_before_every_legacy_mfam_stage() -> None:
    stages = {stage.name: stage for stage in PHASE1_STAGES}
    names = list(stages)

    assert stages["b1_b"].dependencies == ("b1_a",)
    assert stages["m7_gate"].dependencies == ("b1_b",)
    assert names.index("m7_gate") < names.index("smoke_m7")
    assert names.index("smoke_m7") < names.index("formal_m7")
    assert names.index("formal_m7") < names.index("smoke_m0")
    assert all(names.index("formal_m7") < names.index(stage) for stage in ("smoke_m0", "formal_m0"))


def test_b0_reference_is_inspected_after_training_and_before_validation() -> None:
    names = [stage.name for stage in PHASE1_STAGES]

    assert names.index("formal_m3") < names.index("baseline_b0")
    assert names.index("baseline_b0") < names.index("val_all")


def test_workflow_reuses_only_hash_valid_completed_stage(tmp_path: Path) -> None:
    calls: list[str] = []

    def action() -> StageResult:
        calls.append("audit")
        return StageResult(output_hashes={"manifest": "m1"})

    workflow = PipelineWorkflow(tmp_path, pipeline_id="p1", common_input_hashes={"config": "c1"})
    workflow.run_stage("audit", action)
    workflow.run_stage("audit", action)

    assert calls == ["audit"]

    changed = PipelineWorkflow(tmp_path, pipeline_id="p1", common_input_hashes={"config": "c2"})
    changed.run_stage("audit", action)
    assert calls == ["audit", "audit"]


def test_stale_running_smoke_m7_is_restarted_with_incremented_attempt(tmp_path: Path) -> None:
    calls: list[str] = []
    workflow = PipelineWorkflow(
        tmp_path,
        pipeline_id="p1",
        common_input_hashes={"config": "c1"},
    )
    names = [stage.name for stage in PHASE1_STAGES]
    for name in names[: names.index("smoke_m7")]:
        workflow.run_stage(name, lambda name=name: StageResult({name: f"{name}-hash"}))
    stale = PipelineState(
        pipeline_id="p1",
        stage="smoke_m7",
        status="running",
        attempt=4,
        epoch=None,
        input_hashes={"config": "c1", "m7_gate:m7_gate": "m7_gate-hash"},
        output_hashes={},
    )
    smoke_path = tmp_path / "stages" / "smoke_m7.json"
    smoke_path.write_text(json.dumps(stale.to_dict()))

    workflow.run_stage(
        "smoke_m7",
        lambda: (calls.append("smoke_m7") or StageResult({"canonical": "new-hash"})),
    )
    recovered = PipelineState.from_dict(json.loads(smoke_path.read_text()))

    assert calls == ["smoke_m7"]
    assert recovered.attempt == stale.attempt + 1
    assert recovered.status == "completed"
    assert recovered.output_hashes == {"canonical": "new-hash"}


def test_workflow_refuses_stage_before_dependencies(tmp_path: Path) -> None:
    workflow = PipelineWorkflow(tmp_path, pipeline_id="p1", common_input_hashes={"config": "c1"})

    with pytest.raises(RuntimeError, match="dependency"):
        workflow.run_stage("verify", lambda: StageResult({"environment": "e1"}))


def test_test_stage_requires_frozen_selection_artifact(tmp_path: Path) -> None:
    workflow = PipelineWorkflow(tmp_path, pipeline_id="p1", common_input_hashes={"config": "c1"})
    names = [stage.name for stage in PHASE1_STAGES]
    for name in names[: names.index("test_all")]:
        workflow.run_stage(name, lambda name=name: StageResult({name: f"{name}-hash"}))

    with pytest.raises(RuntimeError, match="selection.json"):
        workflow.run_stage("test_all", lambda: StageResult({"test": "t1"}))
