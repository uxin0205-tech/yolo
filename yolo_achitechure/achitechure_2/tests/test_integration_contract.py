from __future__ import annotations

import os

import pytest

from achitechure_2.candidate import build_candidate
from achitechure_2.full35_adapter import (
    CANDIDATE_MODULE_PATHS,
    Full35Release,
)


@pytest.mark.integration
def test_formal_handoff_cpu_validation_is_explicitly_opt_in() -> None:
    """明確 opt-in 時，從 immutable Full35 J3 真正載入並建構 C1。"""

    if os.environ.get("ARCHITECHURE_2_HANDOFF") != "1":
        pytest.skip("設定 ARCHITECHURE_2_HANDOFF=1 才重跑 2.35 GB Full35 materialization")

    release = Full35Release()
    layout = release.verify_layout()
    parent = release.load_parent()
    contract = parent.model.contract()

    assert layout["accepted_stage"] == "j3"
    assert layout["checkpoint_sha256"].startswith("d67fb45c")
    assert parent.checkpoint_report["tensors"] == 1238
    assert contract["model_kind"] == "graph_shared_dual_head"
    assert contract["detect_nc"] == 80
    assert contract["pose_nc"] == 2
    assert contract["kpt_shape"] == [2, 3]

    candidate, build = build_candidate(
        parent.model,
        release.resolved_candidate("C1"),
        seed=0,
    )

    assert build.changed_fields == ("e",)
    assert build.changed_module_paths == CANDIDATE_MODULE_PATHS
    assert build.parent_unchanged
    assert build.model_contract_unchanged
    assert candidate.contract() == contract
