from __future__ import annotations

import pytest
import torch

from yolo_quantize import (
    Full35ActivationAdapter,
    Full35ActivationPolicy,
    Full35WeightViewAdapter,
)
from yolo_quantize.integer_boundaries import (
    IntegerBoundaryContract,
    IntegerTensorSpec,
)


def test_requantize_uses_nearest_even_and_clamps_to_target_codes() -> None:
    contract = IntegerBoundaryContract()
    source = IntegerTensorSpec(bits=8, signed=True, scale=0.25, zero_point=0)
    target = IntegerTensorSpec(bits=4, signed=True, scale=0.5, zero_point=0)

    result = contract.requantize(
        torch.tensor([-128, -3, -1, 0, 1, 3, 127]),
        source=source,
        target=target,
    )

    assert result.codes.tolist() == [-8, -2, 0, 0, 0, 2, 7]
    assert result.saturated_count == 2
    assert result.rounding == "nearest_even"
    assert result.saturation == "clamp"


def test_requantize_preserves_a_non_integral_lsq_plus_offset() -> None:
    contract = IntegerBoundaryContract()
    source = IntegerTensorSpec(
        bits=8,
        signed=False,
        scale=0.25,
        offset=-0.125,
    )
    target = IntegerTensorSpec(
        bits=8,
        signed=True,
        scale=0.25,
        offset=0.125,
    )

    result = contract.requantize(
        torch.tensor([0, 2]),
        source=source,
        target=target,
    )

    assert result.codes.tolist() == [-1, 1]
    assert result.dequantized.tolist() == pytest.approx([-0.125, 0.375])


def test_add_requantizes_both_branches_to_the_registered_output_scale() -> None:
    contract = IntegerBoundaryContract()
    left = IntegerTensorSpec(bits=8, signed=True, scale=0.25, zero_point=0)
    right = IntegerTensorSpec(bits=8, signed=False, scale=0.5, zero_point=2)
    output = IntegerTensorSpec(bits=8, signed=True, scale=0.25, zero_point=0)

    result = contract.add(
        torch.tensor([4, -4]),
        left=left,
        right_codes=torch.tensor([4, 0]),
        right=right,
        output=output,
    )

    assert result.codes.tolist() == [8, -8]
    assert result.accumulator_codes.tolist() == [8, -8]
    assert result.saturated_count == 0


def test_add_rounds_each_aligned_branch_before_integer_accumulation() -> None:
    contract = IntegerBoundaryContract()
    source = IntegerTensorSpec(bits=8, signed=True, scale=0.5, zero_point=0)
    output = IntegerTensorSpec(bits=8, signed=True, scale=1.0, zero_point=0)

    result = contract.add(
        torch.tensor([1]),
        left=source,
        right_codes=torch.tensor([1]),
        right=source,
        output=output,
    )

    assert result.codes.tolist() == [0]
    assert result.accumulator_codes.tolist() == [0]


def test_concat_requantizes_every_branch_before_concatenation() -> None:
    contract = IntegerBoundaryContract()
    output = IntegerTensorSpec(bits=8, signed=True, scale=0.25, zero_point=0)

    result = contract.concat(
        (
            (
                torch.tensor([[2, 4]]),
                IntegerTensorSpec(
                    bits=8,
                    signed=False,
                    scale=0.5,
                    zero_point=0,
                ),
            ),
            (
                torch.tensor([[-2, 2]]),
                IntegerTensorSpec(
                    bits=8,
                    signed=True,
                    scale=0.25,
                    zero_point=0,
                ),
            ),
        ),
        output=output,
        dim=1,
    )

    assert result.codes.tolist() == [[4, 8, -2, 2]]
    assert result.branch_saturated_counts == (0, 0)


def test_accumulator_audit_reports_the_worst_centered_int32_bound() -> None:
    contract = IntegerBoundaryContract(accumulator_bits=32)
    activation = IntegerTensorSpec(
        bits=8,
        signed=False,
        scale=0.125,
        zero_point=128,
    )
    weight = IntegerTensorSpec(bits=8, signed=True, scale=0.01, zero_point=0)

    audit = contract.audit_accumulator(
        reduction_elements=3 * 3 * 256,
        activation=activation,
        weight=weight,
    )

    assert audit.worst_case_absolute == 3 * 3 * 256 * 128 * 128
    assert audit.int32_safe is True
    assert audit.bias_scale_rule == "input_scale_times_weight_scale"
    assert audit.bias_range_check == "deferred_until_calibrated_scales"


def test_full35_boundary_manifest_is_bound_to_the_real_deployment_graph() -> None:
    checkpoint_sha256 = (
        "7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e"
    )
    built = Full35ActivationAdapter().build(
        Full35ActivationPolicy(activation="qsilu_pq", bits=8),
        checkpoint=(
            "/home/uxin/yolo/yolo_activation/artifacts/runs/full35/"
            "short-recovery-v2-lr01-uniform-qsilu-pq-seed1/"
            "inference/best_joint.pt"
        ),
        checkpoint_sha256=checkpoint_sha256,
    )
    views = Full35WeightViewAdapter().build(built.model)

    manifest = IntegerBoundaryContract().inspect_full35(
        views.deployment,
        activation_policy_id=built.policy.policy_id,
        checkpoint_sha256=checkpoint_sha256,
        deployment_state_sha256=views.manifest.deployment_state_sha256,
    )

    assert manifest.counts == {
        "deployment_weight_sites": 148,
        "protected_binary_qk_weight_sites": 4,
        "activation_output_quantizers": 124,
        "top_level_concat_modules": 4,
        "reviewed_core_concat_operations": 21,
        "reviewed_core_residual_add_operations": 20,
        "masf_islands": 1,
        "attention_islands": 2,
        "binary_score_islands": 2,
        "pwl_normalizer_islands": 2,
        "task_head_output_islands": 2,
    }
    assert len(manifest.accumulator_audits) == 148
    assert all(item.int32_safe for item in manifest.accumulator_audits)
    assert manifest.cpu_contract_passed is True
    assert manifest.ready_for_gpu_calibration is True
    assert manifest.native_integer_kernel_claimed is False
    assert manifest.gpu_used is False
    assert manifest.checkpoint_sha256 == checkpoint_sha256
    assert manifest.deployment_state_sha256 == views.manifest.deployment_state_sha256
    assert manifest.activation_affine_policy == (
        "learned_scale_plus_arbitrary_offset_requires_calibrated_bias_lowering"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"bits": 1, "signed": True, "scale": 1.0, "zero_point": 0}, "bits"),
        ({"bits": 8, "signed": True, "scale": 0.0, "zero_point": 0}, "scale"),
        (
            {"bits": 8, "signed": False, "scale": 1.0, "zero_point": 256},
            "zero_point",
        ),
        (
            {
                "bits": 8,
                "signed": False,
                "scale": 1.0,
                "zero_point": 0,
                "offset": 0.25,
            },
            "exactly one",
        ),
    ),
)
def test_integer_tensor_spec_rejects_invalid_contracts(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        IntegerTensorSpec(**kwargs)  # type: ignore[arg-type]
