from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_quantize.activation_policy_search import (
    Full35ActivationPolicySearchPlan,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = PROJECT_ROOT / "configs/experiments/v12-activation-policy-map50-search-v1.yaml"


def test_activation_policy_plan_pins_q3_order_and_total_map50_gate() -> None:
    plan = Full35ActivationPolicySearchPlan.from_yaml(PLAN)

    assert plan.gate_spec.metric_family == "map50"
    assert plan.gate_spec.total_max_drop == 0.015
    assert tuple(candidate.candidate_id for candidate in plan.candidates) == (
        "qsilu-hswish-neck-attention",
        "qsilu-hswish-neck-attention-masf",
        "qsilu-hswish-neck-attention-masf-backbone-attention",
        "uniform-hardswish",
        "uniform-poly-shift",
    )
    assert plan.candidates[0].policy.region_assignments == (
        ("neck_attention", "hardswish"),
    )
    assert plan.candidates[2].checkpoint == plan.candidates[0].checkpoint
    assert plan.formal_training is False
    assert plan.formal_validation is False


def test_activation_policy_list_only_does_not_require_cuda(capsys) -> None:
    result = main(["--plan", str(PLAN), "--list-only"])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["candidate_count"] == 5
    assert payload["q3_evidence"]["sha256"] == (
        "8c495d71a7d4e50db7dd29676dfc8919a88cac3e362ab3f7d4737428c9b41d77"
    )


def test_activation_policy_execution_requires_explicit_acknowledgement(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--plan", str(PLAN)])

    assert "--execute-reviewed-plan" in capsys.readouterr().err
