from __future__ import annotations

from pathlib import Path

from yolo_quantize import Full35ActivationAdapter, Full35ActivationPolicy


def test_full35_adapter_wraps_reviewed_sites_and_preserves_special_modules() -> None:
    adapter = Full35ActivationAdapter()
    policy = Full35ActivationPolicy(activation="qsilu_pq", bits=8)

    preflight = adapter.preflight()
    built = adapter.build(policy)

    assert preflight.ready
    assert built.policy.policy_id == "qsilu_pq--lsq-plus-a8"
    assert built.applied.quantizer_count == 124
    assert len(built.training_only_paths) == 66
    assert built.training_only_regions == (
        "detect_one2many",
        "pose_flow",
        "pose_one2many",
    )
    assert built.applied.mode == "observe"

    contract = built.model.contract()
    assert contract["model_kind"] == "graph_shared_dual_head"
    assert contract["shared_layers"] == 23
    assert contract["head_inputs"] == [16, 19, 22]
    assert contract["strides"] == [8.0, 16.0, 32.0]
    assert contract["reg_max"] == 1
    assert contract["end2end"] is True
    assert contract["detect_nc"] == 80
    assert contract["pose_nc"] == 2
    assert contract["kpt_shape"] == [2, 3]
    assert contract["pose_flow_module"] == "RealNVP"

    assert built.protected_before == built.protected_after
    assert built.protected_after == {
        "achitechure_1.masf.P3MASFFull35": 1,
        "ultralytics.nn.modules.block.RealNVP": 1,
        "ultralytics.nn.modules.head.Detect": 1,
        "ultralytics.nn.modules.head.Pose26": 1,
        "yolo_attention.attention.HardwareFriendlyAttention": 2,
        "yolo_attention.binary_basis.BinaryScore": 2,
        "yolo_attention.normalization.PiecewiseLinearSoftmax": 2,
        "yolo_attention.projection.ModularQKVProjection": 2,
        "yolo_attention.relative_bias.RelativePositionBias": 2,
    }


def test_full35_adapter_loads_an_explicit_activation_parent_checkpoint() -> None:
    checkpoint = Path(
        "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
        "short-recovery-v2-lr01-uniform-hardswish-seed1/"
        "inference/best_joint.pt"
    )
    expected_sha256 = "79e0e4f615a7d8b82da4fd244165d2c035d7a523681833392d30b66e57177731"

    built = Full35ActivationAdapter().build(
        Full35ActivationPolicy(activation="hardswish", bits=8),
        checkpoint=checkpoint,
        checkpoint_sha256=expected_sha256,
    )

    assert built.checkpoint_path == checkpoint
    assert built.checkpoint_sha256 == expected_sha256
    assert built.loaded_checkpoint.path == checkpoint
    assert built.loaded_checkpoint.state_source == "ema"


def test_full35_adapter_keeps_qsilu_parent_and_applies_regional_hardswish() -> None:
    checkpoint = Path(
        "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
        "short-recovery-v2-lr01-uniform-qsilu-pq-seed1/"
        "inference/best_joint.pt"
    )
    expected_sha256 = "7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e"
    policy = Full35ActivationPolicy(
        activation="qsilu_pq",
        bits=8,
        region_assignments=(("neck_attention", "hardswish"),),
    )

    built = Full35ActivationAdapter().build(
        policy,
        checkpoint=checkpoint,
        checkpoint_sha256=expected_sha256,
    )

    assert policy.policy_id == (
        "qsilu_pq--regional--neck_attention=hardswish--lsq-plus-a8"
    )
    assert built.activation_counts == {"hardswish": 1, "qsilu_pq": 189}
    assert built.checkpoint_path == checkpoint


def test_full35_activation_policy_rejects_duplicate_or_unknown_regions() -> None:
    duplicate = (("masf", "hardswish"), ("masf", "hardswish"))

    try:
        Full35ActivationPolicy(
            activation="qsilu_pq",
            bits=8,
            region_assignments=duplicate,
        )
    except ValueError as error:
        assert "duplicate activation region" in str(error)
    else:
        raise AssertionError("duplicate activation regions must fail closed")

    policy = Full35ActivationPolicy(
        activation="qsilu_pq",
        bits=8,
        region_assignments=(("not_a_full35_region", "hardswish"),),
    )
    try:
        Full35ActivationAdapter().build(policy)
    except ValueError as error:
        assert "unknown Full35 activation regions" in str(error)
    else:
        raise AssertionError("unknown activation regions must fail closed")
