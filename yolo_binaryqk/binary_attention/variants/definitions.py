"""Immutable definitions for the corrected YOLO11 BinaryAttention matrix.

T0/T1 are already completed source artifacts.  The formal continuation starts
at T2, then evaluates the three dual-attention branches (T3--T5), applies KD
to the best T1--T5 branch (T6), and finally evaluates the T7 and N4 families.
The materializers keep the selected upstream configuration in every resolved
artifact so that a later audit can reconstruct the exact experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Literal

QKMode = Literal["fp", "sign", "scaled_sign", "dual"]
BiasType = Literal["none", "dense_2d", "decomposed_2d"]


@dataclass(frozen=True)
class VariantDefinition:
    id: str
    attention_type: str
    qk_mode: QKMode
    use_qat: bool
    use_distillation: bool = False
    distillation_type: str | None = None
    kd_components: tuple[str, ...] = ()
    bias_type: BiasType = "none"
    p_bits: int | None = None
    v_bits: int | None = None
    magnitude_bits: int | None = None
    num_binary_qk: int = 0
    num_softmax: int = 1
    num_pv: int = 1
    base_variant: str = "E0"
    kd_target_family: str | None = None

    @property
    def config_hash(self) -> str:
        values = asdict(self)
        # Completed E/T0/T1 artifacts were created before the optional KD
        # scope marker existed.  Keep their hashes backward compatible.
        if values.get("kd_target_family") is None:
            values.pop("kd_target_family", None)
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    @property
    def artifact_name(self) -> str:
        return self.id.replace("/", "-").replace("+", "-")

    @property
    def theoretical_cost(self) -> dict[str, int]:
        return {
            "binary_qk": self.num_binary_qk,
            "softmax": self.num_softmax,
            "pv": self.num_pv,
        }


def _v(id: str, attention_type: str, qk_mode: QKMode, qat: bool = False, **kwargs) -> VariantDefinition:
    return VariantDefinition(id, attention_type, qk_mode, qat, **kwargs)


T6_CANDIDATES = ("T6-O", "T6-F", "T6-A", "T6-O/F", "T6-O/A", "T6-F/A")
T7_VARIANTS = ("T7-D", "T7-R", "T7-P", "T7-V", "T7-PV")
T1_TO_T5 = ("T1", "T2", "T3", "T4", "T5")
NON_KD_BIAS_VARIANTS = frozenset({"N4-FP", "N4-I8", "N4-I4", "N4-PV"})


VARIANTS: dict[str, VariantDefinition] = {
    "E0": _v("E0", "FPAttention", "fp", base_variant="E0"),
    "E1-S": _v("E1-S", "SignOnlyBinaryAttention", "sign", num_binary_qk=1),
    "E1": _v("E1", "ScaledBinaryAttention", "scaled_sign", num_binary_qk=1),
    "E2-DUAL": _v(
        "E2-DUAL", "ResidualDualMatchedBasisAttention", "dual", False,
        num_binary_qk=2, num_softmax=1, num_pv=1, base_variant="E0",
    ),
    "T0": _v("T0", "FPAttention", "fp", base_variant="E0"),
    "T1": _v("T1", "SignOnlyBinaryAttention", "sign", True, num_binary_qk=1, base_variant="T0"),
    "T2": _v("T2", "ScaledBinaryAttention", "scaled_sign", True, num_binary_qk=1, base_variant="T1"),

    # Corrected N1/N2/N3 placement: these are now the formal T3/T4/T5 runs.
    "T3": _v(
        "T3", "ParallelDualBinaryAttention", "dual", True,
        num_binary_qk=2, num_softmax=2, num_pv=2, base_variant="T2",
    ),
    "T4": _v(
        "T4", "ResidualDualFullBasisAttention", "dual", True,
        num_binary_qk=4, num_softmax=1, num_pv=1, base_variant="T3",
    ),
    "T5": _v(
        "T5", "ResidualDualMatchedBasisAttention", "dual", True,
        num_binary_qk=2, num_softmax=1, num_pv=1, base_variant="T4",
    ),

    # T6 is materialized from whichever T1--T5 full run has the best mAP.
    "T6-O": _v(
        "T6-O", "ScaledBinaryAttention", "scaled_sign", True,
        use_distillation=True, distillation_type="positional_encoding",
        kd_components=("positional",), base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T6-F": _v(
        "T6-F", "ScaledBinaryAttention", "scaled_sign", True,
        use_distillation=True, distillation_type="feature",
        kd_components=("feature",), base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T6-A": _v(
        "T6-A", "ScaledBinaryAttention", "scaled_sign", True,
        use_distillation=True, distillation_type="attention",
        kd_components=("attention",), base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T6-O/F": _v(
        "T6-O/F", "ScaledBinaryAttention", "scaled_sign", True,
        use_distillation=True, distillation_type="positional_encoding+feature",
        kd_components=("positional", "feature"), base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T6-O/A": _v(
        "T6-O/A", "ScaledBinaryAttention", "scaled_sign", True,
        use_distillation=True, distillation_type="positional_encoding+attention",
        kd_components=("positional", "attention"), base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T6-F/A": _v(
        "T6-F/A", "ScaledBinaryAttention", "scaled_sign", True,
        use_distillation=True, distillation_type="feature+attention",
        kd_components=("feature", "attention"), base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T6": _v(
        "T6", "ScaledBinaryAttention", "scaled_sign", True,
        use_distillation=True, distillation_type="selected", kd_components=(),
        base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),

    # T7-D/R select the relative-position parameterization; T7-P/V/PV add
    # fake quantization to the selected T7 bias branch.
    "T7-D": _v(
        "T7-D", "ScaledBinaryAttention", "scaled_sign", True,
        bias_type="dense_2d", base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T7-R": _v(
        "T7-R", "ScaledBinaryAttention", "scaled_sign", True,
        bias_type="decomposed_2d", base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T7-P": _v(
        "T7-P", "ScaledBinaryAttention", "scaled_sign", True,
        p_bits=8, base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T7-V": _v(
        "T7-V", "ScaledBinaryAttention", "scaled_sign", True,
        v_bits=8, base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),
    "T7-PV": _v(
        "T7-PV", "ScaledBinaryAttention", "scaled_sign", True,
        p_bits=8, v_bits=8, base_variant="best-T1-T5", kd_target_family="T1-T5",
    ),

    # N4 is always non-KD.  Its parent/bias and selected magnitude mode are
    # materialized after the best non-KD T1--T5/T7 candidate is known.
    "N4-FP": _v(
        "N4-FP", "MagnitudeSideChannelAttention", "dual", True,
        num_binary_qk=2, num_softmax=1, num_pv=1, base_variant="best-non-KD(T1-T5+T7)",
    ),
    "N4-I8": _v(
        "N4-I8", "MagnitudeSideChannelAttention", "dual", True,
        magnitude_bits=8, num_binary_qk=2, num_softmax=1, num_pv=1, base_variant="best-non-KD(T1-T5+T7)",
    ),
    "N4-I4": _v(
        "N4-I4", "MagnitudeSideChannelAttention", "dual", True,
        magnitude_bits=4, num_binary_qk=2, num_softmax=1, num_pv=1, base_variant="best-non-KD(T1-T5+T7)",
    ),
    "N4-PV": _v(
        "N4-PV", "MagnitudeSideChannelAttention", "dual", True,
        p_bits=8, v_bits=8, num_binary_qk=2, num_softmax=1, num_pv=1,
        base_variant="best-non-KD(T1-T5+T7)",
    ),
}

# Shell-friendly aliases for pairwise KD names.
VARIANTS["T6-OF"] = VARIANTS["T6-O/F"]
VARIANTS["T6-OA"] = VARIANTS["T6-O/A"]
VARIANTS["T6-FA"] = VARIANTS["T6-F/A"]

KD_INHERITING_VARIANTS = frozenset(T7_VARIANTS)
BIAS_INHERITING_VARIANTS = frozenset({"T7-P", "T7-V", "T7-PV"})
_KD_COMPONENTS = {"positional", "output", "feature", "attention", "hard"}
_INHERITED_FIELDS = (
    "attention_type", "qk_mode", "use_qat", "bias_type", "p_bits", "v_bits", "magnitude_bits",
    "num_binary_qk", "num_softmax", "num_pv",
)


def get_variant(variant_id: str) -> VariantDefinition:
    try:
        return VARIANTS[variant_id]
    except KeyError as exc:
        raise ValueError(f"Unknown variant '{variant_id}'. Available: {', '.join(VARIANTS)}") from exc


def quantization_contract(variant: VariantDefinition) -> dict[str, str]:
    """Return auditable tensor-axis and initialization semantics."""

    return {
        "qk_scale": (
            "mean_abs_channel_token_per_sample_head"
            if variant.qk_mode == "scaled_sign"
            else "residual_basis_channel_per_token"
            if variant.qk_mode == "dual"
            else "none"
        ),
        "p8_scale": "static_unsigned_1_over_255" if variant.p_bits == 8 else "none",
        "v8_scale": "max_abs_token_per_sample_head_channel" if variant.v_bits == 8 else "none",
        "bias_parameterization": {
            "none": "none",
            "dense_2d": "full_2d_relative_position",
            "decomposed_2d": "axis_decomposed_relative_position",
        }[variant.bias_type],
        "bias_initialization": (
            "truncated_normal_std_0.02"
            if variant.bias_type == "dense_2d"
            else "zeros"
            if variant.bias_type == "decomposed_2d"
            else "none"
        ),
    }


def _components(value: tuple[str, ...] | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(value if value else fallback))
    if not result or any(item not in _KD_COMPONENTS for item in result):
        raise ValueError(f"invalid KD components: {result}")
    return result


def _copy_upstream(template: VariantDefinition, upstream: VariantDefinition, *, base_variant: str | None = None) -> dict:
    values = asdict(template)
    for key in _INHERITED_FIELDS:
        values[key] = getattr(upstream, key)
    values["base_variant"] = base_variant or upstream.id
    return values


def materialize_t6_candidate(
    variant_id: str,
    *,
    base_variant: str,
    components: tuple[str, ...] | None = None,
) -> VariantDefinition:
    """Apply one T6 KD choice to the selected best T1--T5 architecture."""

    template = get_variant(variant_id)
    if template.id not in T6_CANDIDATES and template.id != "T6":
        raise ValueError(f"{template.id} is not a T6 candidate/selection")
    upstream = get_variant(base_variant)
    if upstream.id not in T1_TO_T5:
        raise ValueError(f"T6 base must be one of T1--T5, got {base_variant}")
    selected = _components(components, template.kd_components)
    values = _copy_upstream(template, upstream, base_variant=upstream.id)
    values.update(
        use_distillation=True,
        distillation_type="+".join(selected),
        kd_components=selected,
        kd_target_family="T1-T5",
    )
    return VariantDefinition(**values)


def materialize_selected_t6(base_variant: str, components: tuple[str, ...]) -> VariantDefinition:
    return materialize_t6_candidate("T6", base_variant=base_variant, components=components)


def materialize_t7_variant(
    variant_id: str,
    *,
    base_variant: str,
    kd_components: tuple[str, ...],
    bias_type: BiasType | None = None,
) -> VariantDefinition:
    """Materialize a T7 variant from the selected T6 base and KD choice."""

    template = get_variant(variant_id)
    if template.id not in T7_VARIANTS:
        raise ValueError(f"{template.id} is not a T7 variant")
    selected_t6 = materialize_selected_t6(base_variant, kd_components)
    values = _copy_upstream(template, selected_t6, base_variant=selected_t6.base_variant)
    if template.id in {"T7-D", "T7-R"}:
        values["bias_type"] = template.bias_type
    elif bias_type not in {"dense_2d", "decomposed_2d"}:
        raise ValueError(f"{template.id} requires selected T7 bias")
    else:
        values["bias_type"] = bias_type
    # T7-P/V/PV introduce only the requested P/V fake quantizer.
    values["p_bits"] = template.p_bits
    values["v_bits"] = template.v_bits
    values.update(
        use_distillation=True,
        distillation_type="+".join(selected_t6.kd_components),
        kd_components=selected_t6.kd_components,
        kd_target_family="T1-T5",
    )
    return VariantDefinition(**values)


def materialize_non_kd_bias_variant(
    variant_id: str,
    *,
    bias_type: BiasType | None = None,
    magnitude_mode: str | None = None,
    parent_variant: str | None = None,
) -> VariantDefinition:
    """Materialize N4 with no KD and the selected T1--T5/T7 provenance."""

    base = get_variant(variant_id)
    if base.id not in NON_KD_BIAS_VARIANTS:
        raise ValueError(f"{base.id} is not a non-KD N4 variant")
    if bias_type not in {"none", "dense_2d", "decomposed_2d"}:
        raise ValueError(f"{base.id} requires selected bias type")
    values = asdict(base)
    values.update(
        use_distillation=False,
        distillation_type=None,
        kd_components=(),
        kd_target_family=None,
        bias_type=bias_type,
        base_variant=parent_variant or base.base_variant,
    )
    if base.id == "N4-PV":
        mode = magnitude_mode or "fp"
        values["magnitude_bits"] = {"fp": None, "int8": 8, "int4": 4}.get(mode)
        if mode not in {"fp", "int8", "int4"}:
            raise ValueError("N4-PV requires magnitude mode fp, int8 or int4")
    return VariantDefinition(**values)


def variant_from_resolved_config(resolved: dict) -> VariantDefinition:
    """Reconstruct the immutable definition represented by a run artifact."""

    variant_id = str(resolved.get("id", ""))
    components = tuple(resolved.get("kd_components") or ())
    if variant_id in T6_CANDIDATES or variant_id == "T6":
        return materialize_t6_candidate(variant_id, base_variant=str(resolved.get("base_variant")), components=components)
    if variant_id in KD_INHERITING_VARIANTS:
        return materialize_t7_variant(
            variant_id,
            base_variant=str(resolved.get("base_variant")),
            kd_components=components,
            bias_type=resolved.get("bias_type"),
        )
    if variant_id in NON_KD_BIAS_VARIANTS:
        magnitude = resolved.get("magnitude_bits")
        mode = {None: "fp", 8: "int8", 4: "int4"}.get(magnitude)
        return materialize_non_kd_bias_variant(
            variant_id,
            bias_type=resolved.get("bias_type"),
            magnitude_mode=mode,
            parent_variant=resolved.get("base_variant"),
        )
    return get_variant(variant_id)


# Compatibility helpers retained for older callers; the corrected plan uses
# the explicit T6 names above.
def materialize_selected_t3(components: tuple[str, ...]) -> VariantDefinition:
    return materialize_selected_t6("T2", components)


def materialize_inherited_variant(
    variant_id: str,
    *,
    kd_components: tuple[str, ...],
    bias_type: BiasType | None = None,
    magnitude_mode: str | None = None,
) -> VariantDefinition:
    return materialize_t7_variant(
        variant_id,
        base_variant="T2",
        kd_components=kd_components,
        bias_type=bias_type,
    )


def select_best_t6_variant(results: list[dict], base_variant: str) -> VariantDefinition:
    """Select a T6 component combination by recorded AP50--95."""

    candidates = []
    for row in results:
        candidate = VARIANTS.get(str(row.get("variant", "")))
        score = row.get("mAP50_95")
        if candidate is None or candidate.id not in T6_CANDIDATES or not candidate.kd_components:
            continue
        if isinstance(score, (int, float)):
            candidates.append((float(score), candidate.kd_components))
    if not candidates:
        raise ValueError("no completed T6 candidate with numeric mAP50_95")
    _, components = max(candidates, key=lambda item: item[0])
    return materialize_selected_t6(base_variant, components)


def select_best_t3_variant(results: list[dict]) -> VariantDefinition:
    """Legacy name; select the T6 combination against T2 for old callers."""

    return select_best_t6_variant(results, "T2")
