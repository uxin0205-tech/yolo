from __future__ import annotations

from masf_yolo.artifacts.state import can_reuse_stage
from masf_yolo.contracts import PipelineState


def _complete_state() -> PipelineState:
    return PipelineState(
        pipeline_id="pipeline-a",
        stage="audit",
        status="completed",
        attempt=1,
        epoch=None,
        input_hashes={"config": "c1", "data": "d1"},
        output_hashes={"manifest": "m1"},
    )


def test_completed_stage_reuse_requires_all_exact_hashes() -> None:
    state = _complete_state()

    assert can_reuse_stage(state, {"config": "c1", "data": "d1"}, {"manifest": "m1"})
    assert not can_reuse_stage(state, {"config": "changed", "data": "d1"}, {"manifest": "m1"})
    assert not can_reuse_stage(state, {"config": "c1", "data": "d1"}, {"manifest": "changed"})


def test_running_or_failed_stage_is_never_reused() -> None:
    complete = _complete_state()
    running = PipelineState.from_dict({**complete.to_dict(), "status": "running"})
    failed = PipelineState.from_dict({**complete.to_dict(), "status": "failed"})

    assert not can_reuse_stage(running, running.input_hashes, running.output_hashes)
    assert not can_reuse_stage(failed, failed.input_hashes, failed.output_hashes)
