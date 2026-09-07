"""CPU reference semantics for hybrid-integer Full35 graph boundaries."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .weight_quantization import Full35WeightRegionCatalog


@dataclass(frozen=True)
class IntegerTensorSpec:
    """One affine integer tensor contract with a frozen numeric scale."""

    bits: int
    signed: bool
    scale: float
    zero_point: int | None = None
    offset: float | None = None

    def __post_init__(self) -> None:
        if self.bits < 2 or self.bits > 32:
            raise ValueError("integer tensor bits must be in [2, 32]")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("integer tensor scale must be finite and positive")
        if (self.zero_point is None) == (self.offset is None):
            raise ValueError(
                "integer tensor requires exactly one of zero_point or offset"
            )
        if self.zero_point is not None:
            if self.zero_point < self.qmin or self.zero_point > self.qmax:
                raise ValueError("integer tensor zero_point is outside the code range")
        elif self.offset is None or not math.isfinite(self.offset):
            raise ValueError("integer tensor offset must be finite")

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    @property
    def maximum_centered_code(self) -> int:
        if self.zero_point is None:
            raise ValueError(
                "arbitrary affine offset needs calibrated bias lowering before MAC audit"
            )
        return max(abs(self.qmin - self.zero_point), abs(self.qmax - self.zero_point))

    @property
    def real_offset(self) -> float:
        if self.offset is not None:
            return self.offset
        assert self.zero_point is not None
        return -self.zero_point * self.scale


@dataclass(frozen=True)
class RequantizationResult:
    """Codes and audit evidence returned by one reference requantization."""

    codes: Tensor
    dequantized: Tensor
    saturated_count: int
    rounding: str = "nearest_even"
    saturation: str = "clamp"


@dataclass(frozen=True)
class AddResult:
    """Registered-scale result of an integer residual addition."""

    codes: Tensor
    dequantized: Tensor
    accumulator_codes: Tensor
    saturated_count: int


@dataclass(frozen=True)
class ConcatResult:
    """Registered-scale result of an integer concatenation boundary."""

    codes: Tensor
    dequantized: Tensor
    branch_saturated_counts: tuple[int, ...]


@dataclass(frozen=True)
class AccumulatorAudit:
    """Static worst-case centered-code audit for one MAC reduction."""

    reduction_elements: int
    worst_case_absolute: int
    accumulator_bits: int
    signed_maximum: int
    int32_safe: bool
    bias_scale_rule: str = "input_scale_times_weight_scale"
    bias_range_check: str = "deferred_until_calibrated_scales"


@dataclass(frozen=True)
class AccumulatorSiteAudit:
    """One Full35 deployment weight site's conservative W8/A8 MAC bound."""

    path: str
    region: str
    module_type: str
    reduction_elements: int
    activation_code_absolute_maximum: int
    weight_code_absolute_maximum: int
    worst_case_absolute: int
    signed_int32_maximum: int
    int32_safe: bool
    bias_present: bool
    bias_lowering: str = "deferred_until_calibrated_activation_scale_and_offset"

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "region": self.region,
            "module_type": self.module_type,
            "reduction_elements": self.reduction_elements,
            "activation_code_absolute_maximum": (self.activation_code_absolute_maximum),
            "weight_code_absolute_maximum": self.weight_code_absolute_maximum,
            "worst_case_absolute": self.worst_case_absolute,
            "signed_int32_maximum": self.signed_int32_maximum,
            "int32_safe": self.int32_safe,
            "bias_present": self.bias_present,
            "bias_lowering": self.bias_lowering,
        }


@dataclass(frozen=True)
class Full35IntegerBoundaryManifest:
    """Hash-bound CPU evidence for the Full35 hybrid-integer graph Seam."""

    schema_version: int
    activation_policy_id: str
    checkpoint_sha256: str
    deployment_state_sha256: str
    counts: dict[str, int]
    join_policies: dict[str, str]
    protected_island_policies: dict[str, str]
    accumulator_audits: tuple[AccumulatorSiteAudit, ...]
    activation_affine_policy: str
    cpu_contract_passed: bool
    ready_for_gpu_calibration: bool
    native_integer_kernel_claimed: bool
    gpu_used: bool
    next_gpu_requirements: tuple[str, ...]
    hardware_claim_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "formal_training": False,
            "gpu_used": self.gpu_used,
            "activation_policy_id": self.activation_policy_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "deployment_state_sha256": self.deployment_state_sha256,
            "counts": self.counts,
            "join_policies": self.join_policies,
            "protected_island_policies": self.protected_island_policies,
            "activation_affine_policy": self.activation_affine_policy,
            "accumulator_audits": [item.to_dict() for item in self.accumulator_audits],
            "cpu_contract_passed": self.cpu_contract_passed,
            "ready_for_gpu_calibration": self.ready_for_gpu_calibration,
            "native_integer_kernel_claimed": self.native_integer_kernel_claimed,
            "next_gpu_requirements": list(self.next_gpu_requirements),
            "hardware_claim_blockers": list(self.hardware_claim_blockers),
        }


class IntegerBoundaryContract:
    """Reference Interface for rounding, saturation, joins, and MAC bounds.

    This Module intentionally defines the CPU reference semantics only.  A target
    backend still has to lower each frozen real scale ratio into its multiplier and
    shift representation before claiming native integer or bit-true hardware parity.
    """

    def __init__(self, *, accumulator_bits: int = 32) -> None:
        if accumulator_bits < 2 or accumulator_bits > 64:
            raise ValueError("accumulator_bits must be in [2, 64]")
        self.accumulator_bits = accumulator_bits

    @staticmethod
    def _center(codes: Tensor, spec: IntegerTensorSpec) -> Tensor:
        if not torch.is_tensor(codes):
            raise TypeError("integer boundary codes must be tensors")
        if codes.dtype.is_floating_point or codes.dtype == torch.bool:
            raise TypeError("integer boundary codes must use an integer dtype")
        code64 = codes.to(dtype=torch.int64)
        if bool(((code64 < spec.qmin) | (code64 > spec.qmax)).any()):
            raise ValueError("integer boundary input code is outside its source range")
        return code64 - spec.zero_point

    @staticmethod
    def _finish(
        unbounded_codes: Tensor,
        target: IntegerTensorSpec,
    ) -> RequantizationResult:
        rounded = torch.round(unbounded_codes).to(dtype=torch.int64)
        saturated = (rounded < target.qmin) | (rounded > target.qmax)
        codes = rounded.clamp(target.qmin, target.qmax)
        dequantized = codes.to(torch.float64) * target.scale + target.real_offset
        return RequantizationResult(
            codes=codes,
            dequantized=dequantized,
            saturated_count=int(saturated.sum().item()),
        )

    def requantize(
        self,
        codes: Tensor,
        *,
        source: IntegerTensorSpec,
        target: IntegerTensorSpec,
    ) -> RequantizationResult:
        """Requantize affine codes using RNE and saturating target clamps."""

        source_real = (
            self._center(codes, source).to(torch.float64) * source.scale
            if source.zero_point is not None
            else codes.to(torch.float64) * source.scale + source.real_offset
        )
        target_codes = (source_real - target.real_offset) / target.scale
        return self._finish(target_codes, target)

    def add(
        self,
        left_codes: Tensor,
        *,
        left: IntegerTensorSpec,
        right_codes: Tensor,
        right: IntegerTensorSpec,
        output: IntegerTensorSpec,
    ) -> AddResult:
        """Align two branch scales and add them into one registered output scale."""

        left_real = (
            self._center(left_codes, left).to(torch.float64) * left.scale
            if left.zero_point is not None
            else left_codes.to(torch.float64) * left.scale + left.real_offset
        )
        right_real = (
            self._center(right_codes, right).to(torch.float64) * right.scale
            if right.zero_point is not None
            else right_codes.to(torch.float64) * right.scale + right.real_offset
        )
        left_contribution = torch.round(left_real / output.scale).to(torch.int64)
        right_contribution = torch.round(right_real / output.scale).to(torch.int64)
        if left_contribution.shape != right_contribution.shape:
            raise ValueError("integer add branches must have identical shapes")
        output_origin_code = round(-output.real_offset / output.scale)
        accumulator_codes = left_contribution + right_contribution + output_origin_code
        result = self._finish(accumulator_codes.to(torch.float64), output)
        return AddResult(
            codes=result.codes,
            dequantized=result.dequantized,
            accumulator_codes=accumulator_codes,
            saturated_count=result.saturated_count,
        )

    def concat(
        self,
        branches: tuple[tuple[Tensor, IntegerTensorSpec], ...],
        *,
        output: IntegerTensorSpec,
        dim: int,
    ) -> ConcatResult:
        """Requantize all branches to one registered scale, then concatenate."""

        if not branches:
            raise ValueError("integer concat requires at least one branch")
        requantized = tuple(
            self.requantize(codes, source=source, target=output)
            for codes, source in branches
        )
        codes = torch.cat(tuple(item.codes for item in requantized), dim=dim)
        dequantized = codes.to(torch.float64) * output.scale + output.real_offset
        return ConcatResult(
            codes=codes,
            dequantized=dequantized,
            branch_saturated_counts=tuple(item.saturated_count for item in requantized),
        )

    def audit_accumulator(
        self,
        *,
        reduction_elements: int,
        activation: IntegerTensorSpec,
        weight: IntegerTensorSpec,
    ) -> AccumulatorAudit:
        """Bound centered integer MACs before calibrated bias is available."""

        if reduction_elements <= 0:
            raise ValueError("reduction_elements must be positive")
        worst_case = (
            reduction_elements
            * activation.maximum_centered_code
            * weight.maximum_centered_code
        )
        signed_maximum = (1 << (self.accumulator_bits - 1)) - 1
        return AccumulatorAudit(
            reduction_elements=reduction_elements,
            worst_case_absolute=worst_case,
            accumulator_bits=self.accumulator_bits,
            signed_maximum=signed_maximum,
            int32_safe=worst_case <= signed_maximum,
        )

    def inspect_full35(
        self,
        model: nn.Module,
        *,
        activation_policy_id: str,
        checkpoint_sha256: str,
        deployment_state_sha256: str,
    ) -> Full35IntegerBoundaryManifest:
        """Inspect the reviewed BN-folded Full35 graph without executing a batch."""

        if not activation_policy_id:
            raise ValueError("activation_policy_id must not be empty")
        for label, digest in (
            ("checkpoint_sha256", checkpoint_sha256),
            ("deployment_state_sha256", deployment_state_sha256),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")

        catalog = Full35WeightRegionCatalog.inspect(model)
        if catalog.training_only_sites:
            raise ValueError(
                "integer boundary manifest requires the BN-folded deployment view"
            )
        modules = dict(model.named_modules(remove_duplicate=False))
        class_counts = Counter(
            f"{type(module).__module__}.{type(module).__name__}"
            for module in modules.values()
        )
        core_concat_classes = {
            "achitechure_1.model.C3k2P3MASFFull35",
            "ultralytics.nn.modules.block.C2PSA",
            "ultralytics.nn.modules.block.C3k",
            "ultralytics.nn.modules.block.C3k2",
            "ultralytics.nn.modules.block.SPPF",
            "ultralytics.nn.modules.conv.Concat",
        }
        concat_operations = sum(
            count for name, count in class_counts.items() if name in core_concat_classes
        )
        residual_add_operations = 0
        for module in modules.values():
            if not bool(getattr(module, "add", False)):
                continue
            class_name = f"{type(module).__module__}.{type(module).__name__}"
            if class_name == "ultralytics.nn.modules.block.PSABlock":
                residual_add_operations += 2
            elif class_name in {
                "ultralytics.nn.modules.block.Bottleneck",
                "ultralytics.nn.modules.block.SPPF",
            }:
                residual_add_operations += 1

        counts = {
            "deployment_weight_sites": len(catalog.deployment_sites),
            "protected_binary_qk_weight_sites": len(catalog.protected_sites),
            "activation_output_quantizers": class_counts[
                "yolo_quantize.quantizers.LSQPlusActivationQuantizer"
            ],
            "top_level_concat_modules": class_counts[
                "ultralytics.nn.modules.conv.Concat"
            ],
            "reviewed_core_concat_operations": concat_operations,
            "reviewed_core_residual_add_operations": residual_add_operations,
            "masf_islands": class_counts["achitechure_1.masf.P3MASFFull35"],
            "attention_islands": class_counts[
                "yolo_attention.attention.HardwareFriendlyAttention"
            ],
            "binary_score_islands": class_counts[
                "yolo_attention.binary_basis.BinaryScore"
            ],
            "pwl_normalizer_islands": class_counts[
                "yolo_attention.normalization.PiecewiseLinearSoftmax"
            ],
            "task_head_output_islands": (
                class_counts["ultralytics.nn.modules.head.Detect"]
                + class_counts["ultralytics.nn.modules.head.Pose26"]
            ),
        }
        expected_counts = {
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
        if counts != expected_counts:
            raise RuntimeError(
                "Full35 integer boundary graph signature drifted: "
                f"observed={counts}, expected={expected_counts}"
            )

        signed_int32_maximum = (1 << 31) - 1
        accumulator_audits: list[AccumulatorSiteAudit] = []
        for site in catalog.deployment_sites:
            module = modules[site.path]
            if isinstance(module, nn.Conv2d):
                reduction_elements = (
                    module.weight.shape[1]
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
            elif isinstance(module, nn.Linear):
                reduction_elements = module.in_features
            else:  # pragma: no cover - catalog owns this invariant
                raise TypeError(f"unsupported deployment weight module: {site.path}")
            worst_case = reduction_elements * 255 * 128
            accumulator_audits.append(
                AccumulatorSiteAudit(
                    path=site.path,
                    region=site.region,
                    module_type=site.module_type,
                    reduction_elements=reduction_elements,
                    activation_code_absolute_maximum=255,
                    weight_code_absolute_maximum=128,
                    worst_case_absolute=worst_case,
                    signed_int32_maximum=signed_int32_maximum,
                    int32_safe=worst_case <= signed_int32_maximum,
                    bias_present=module.bias is not None,
                )
            )
        all_mac_bounds_safe = all(item.int32_safe for item in accumulator_audits)
        if not all_mac_bounds_safe:
            unsafe = ", ".join(
                item.path for item in accumulator_audits if not item.int32_safe
            )
            raise RuntimeError(f"Full35 W8/A8 INT32 MAC bound failed: {unsafe}")

        return Full35IntegerBoundaryManifest(
            schema_version=1,
            activation_policy_id=activation_policy_id,
            checkpoint_sha256=checkpoint_sha256,
            deployment_state_sha256=deployment_state_sha256,
            counts=counts,
            join_policies={
                "add": (
                    "requantize_every_input_to_a_registered_output_scale_then_"
                    "int32_add_rne_saturate"
                ),
                "concat": (
                    "requantize_every_branch_to_one_registered_output_scale_"
                    "before_channel_concat"
                ),
                "rounding": "nearest_even",
                "saturation": "clamp_to_declared_code_range",
                "accumulator": "signed_int32",
                "bias": "signed_int32_at_input_scale_times_weight_scale",
            },
            protected_island_policies={
                "masf": "float_or_custom_kernel_with_explicit_qdq_ingress_egress",
                "attention": (
                    "protect_binary_qk_and_pwl_stack_with_explicit_qdq_ingress_egress"
                ),
                "pwl_denominator": "floating_reference_until_integer_reciprocal_exists",
                "task_head_output": (
                    "keep_decode_topk_gather_and_task_output_in_high_precision"
                ),
            },
            accumulator_audits=tuple(accumulator_audits),
            activation_affine_policy=(
                "learned_scale_plus_arbitrary_offset_requires_calibrated_bias_lowering"
            ),
            cpu_contract_passed=True,
            ready_for_gpu_calibration=True,
            native_integer_kernel_claimed=False,
            gpu_used=False,
            next_gpu_requirements=(
                "freeze_each_activation_output_scale_and_offset_on_calibration_data",
                "register_add_and_concat_output_scales_and_measure_saturation",
                "fold_lsq_plus_offset_and_padding_zero_correction_into_int32_bias",
                "compare_every_boundary_reference_code_to_gpu_fake_quant_output",
            ),
            hardware_claim_blockers=(
                "lower_real_scale_ratios_to_target_multiplier_and_shift",
                "replace_floating_pwl_denominator_with_a_reviewed_integer_reciprocal",
                "implement_target_kernels_and_measure_latency_power_and_resources",
            ),
        )
