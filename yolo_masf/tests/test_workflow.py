from __future__ import annotations

from pathlib import Path

import pytest

from masf_yolo.workflow import PHASE1_STAGES, PipelineWorkflow, StageResult


def test_phase1_dag_has_selection_before_test_and_no_phase2_nodes() -> None:
    names = [stage.name for stage in PHASE1_STAGES]

    assert names.index("selection") < names.index("test_all")
    assert names[-2:] == ["final_audit", "report"]
    assert not any("s0" in name.lower() or "temporal" in name.lower() for name in names)
    for index, stage in enumerate(PHASE1_STAGES):
        assert set(stage.dependencies) <= set(names[:index])


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
