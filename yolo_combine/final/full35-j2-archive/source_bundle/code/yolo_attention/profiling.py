"""與 runtime profiling 分離的 analytical operation accounting。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import BasisKind, BDCNDenominator, BDCNProjection, NormalizationKind, VariantConfig


@dataclass(frozen=True)
class AttentionShape:
    tokens: int
    heads: int
    key_dim: int
    value_dim: int
    packed_word_bits: int = 32
    bdcn_levels: int = 16

    def __post_init__(self) -> None:
        if min(self.tokens, self.heads, self.key_dim, self.value_dim, self.bdcn_levels) < 1:
            raise ValueError("attention dimensions must be positive")
        if self.key_dim % self.packed_word_bits:
            raise ValueError("key_dim must be divisible by packed_word_bits")


@dataclass(frozen=True)
class OperationReport:
    fp_qk_mac: int
    pv_mac: int
    single_binary_word_ops: int
    dual_binary_word_ops: int
    hadamard_add_sub: int
    score_entries: int
    bdcn_score_lookups: int
    bdcn_value_bucket_add: int
    bdcn_codebook_value_ops: int
    bdcn_reciprocal_rows: int
    bdcn_materialized_probability_entries: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_operations(shape: AttentionShape) -> OperationReport:
    score_entries = shape.heads * shape.tokens**2
    single_binary = score_entries * (shape.key_dim // shape.packed_word_bits)
    return OperationReport(
        fp_qk_mac=score_entries * shape.key_dim,
        pv_mac=score_entries * shape.value_dim,
        single_binary_word_ops=single_binary,
        dual_binary_word_ops=2 * single_binary,
        hadamard_add_sub=(2 * shape.heads * shape.tokens * shape.key_dim * int(math.log2(shape.key_dim))),
        score_entries=score_entries,
        bdcn_score_lookups=score_entries,
        bdcn_value_bucket_add=score_entries * shape.value_dim,
        bdcn_codebook_value_ops=(shape.heads * shape.tokens * shape.bdcn_levels * shape.value_dim),
        bdcn_reciprocal_rows=shape.heads * shape.tokens,
        bdcn_materialized_probability_entries=0,
    )


def write_variant_profile(
    run_dir: str | Path,
    config: VariantConfig,
    *,
    shape: AttentionShape | None = None,
    attention_sites: int = 2,
) -> Path:
    """寫入可比較的 algorithmic proxies；這些不是實際板端量測。"""

    if attention_sites < 1:
        raise ValueError("attention_sites must be positive")
    reference = shape or AttentionShape(tokens=400, heads=4, key_dim=32, value_dim=64)
    reference = replace(reference, bdcn_levels=config.bdcn_levels)
    operations = estimate_operations(reference)
    counts = {name: value * attention_sites for name, value in operations.to_dict().items()}
    score_entries = counts["score_entries"]
    rows = attention_sites * reference.heads * reference.tokens

    selected_fp_qk_mac = 0
    selected_binary_word_ops = 0
    selected_hadamard_add_sub = 0
    selected_t5_fusion_ops = 0
    if config.basis is BasisKind.FP:
        score_cost = counts["fp_qk_mac"]
        selected_fp_qk_mac = counts["fp_qk_mac"]
    else:
        score_cost = counts["single_binary_word_ops"]
        selected_binary_word_ops = counts["single_binary_word_ops"]
        if config.basis is BasisKind.HADAMARD:
            score_cost = counts["dual_binary_word_ops"]
            score_cost += counts["hadamard_add_sub"]
            selected_binary_word_ops = counts["dual_binary_word_ops"]
            selected_hadamard_add_sub = counts["hadamard_add_sub"]
        elif config.basis is BasisKind.T5:
            # T5 residual approximation 會增加一次 binary-score fusion pass。
            score_cost = counts["dual_binary_word_ops"]
            score_cost += score_entries
            selected_binary_word_ops = counts["dual_binary_word_ops"]
            selected_t5_fusion_ops = score_entries

    if config.normalization is NormalizationKind.BDCN:
        value_cost = counts["bdcn_value_bucket_add"]
        projection_factor = {
            BDCNProjection.FLOAT: 2,
            BDCNProjection.ONE_POT: 1,
            BDCNProjection.TWO_POT: 2,
        }[config.bdcn_projection]
        normalization_cost = (
            counts["bdcn_score_lookups"] + projection_factor * counts["bdcn_codebook_value_ops"]
        )
        denominator_factor = {
            BDCNDenominator.EXACT: 10,
            BDCNDenominator.RECIPROCAL_LUT: 3,
            BDCNDenominator.POT_SHIFT: 1,
        }[config.bdcn_denominator]
        normalization_cost += denominator_factor * counts["bdcn_reciprocal_rows"]
        if config.bdcn_reciprocal_newton_steps:
            normalization_cost += 2 * counts["bdcn_reciprocal_rows"]
        memory_traffic = score_entries + (
            attention_sites * reference.heads * reference.tokens * reference.bdcn_levels * reference.value_dim
        )
    else:
        value_cost = counts["pv_mac"]
        per_entry = {
            NormalizationKind.EXACT: 5,
            NormalizationKind.LUT: 2,
            NormalizationKind.INTEGER_LUT: 2,
            NormalizationKind.PIECEWISE_LINEAR: 3,
            NormalizationKind.BIT_TRUE_PWL: 3,
            NormalizationKind.POWER_OF_TWO: 1,
            NormalizationKind.HARD_SIGMOID: 2,
            NormalizationKind.RELU: 1,
            NormalizationKind.MULTIMAX: 2,
        }[config.normalization]
        normalization_cost = per_entry * score_entries + 2 * rows
        memory_traffic = 2 * score_entries + (
            attention_sites * reference.heads * reference.tokens * reference.value_dim
        )

    payload = {
        "schema_version": 1,
        "kind": "analytical_proxy",
        "variant": config.name,
        "estimated_memory_traffic": float(memory_traffic),
        "arithmetic_cost_proxy": float(score_cost + value_cost + normalization_cost),
        "selected_operation_counts": {
            "fp_qk_mac": selected_fp_qk_mac,
            "binary_word_ops": selected_binary_word_ops,
            "hadamard_add_sub": selected_hadamard_add_sub,
            "t5_fusion_ops": selected_t5_fusion_ops,
            "value_path_ops": value_cost,
            "normalization_ops": normalization_cost,
        },
        "operation_counts": counts,
        "assumptions": {
            "attention_sites": attention_sites,
            "tokens_per_site": reference.tokens,
            "heads": reference.heads,
            "key_dim": reference.key_dim,
            "value_dim": reference.value_dim,
            "bdcn_levels": reference.bdcn_levels,
            "hardware_measurement": False,
            "units": "comparable algorithmic proxy units",
        },
    }
    destination = Path(run_dir) / "profiles" / "analytical.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination.resolve()
