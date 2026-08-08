"""Pipeline state reuse gates."""

from __future__ import annotations

from typing import Mapping

from masf_yolo.contracts import PipelineState


def can_reuse_stage(
    state: PipelineState,
    expected_input_hashes: Mapping[str, str],
    observed_output_hashes: Mapping[str, str],
) -> bool:
    return (
        state.status == "completed"
        and state.input_hashes == dict(expected_input_hashes)
        and state.output_hashes == dict(observed_output_hashes)
    )
