from __future__ import annotations

import json

from yolo_attention.config import (
    BDCNCodebookKind,
    BDCNDenominator,
    BDCNProjection,
    BDCNSharing,
    NormalizationKind,
    VariantConfig,
)
from yolo_attention.profiling import write_variant_profile


def test_profile_writes_required_final_selection_cost_fields(tmp_path) -> None:
    config = VariantConfig(name="I")

    path = write_variant_profile(tmp_path, config)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["kind"] == "analytical_proxy"
    assert payload["estimated_memory_traffic"] > 0
    assert payload["arithmetic_cost_proxy"] > 0
    assert "operation_counts" in payload


def test_fused_bdcn_profile_removes_dense_probability_materialization(tmp_path) -> None:
    config = VariantConfig(
        name="D2-1P",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.ONE_POT,
        bdcn_denominator=BDCNDenominator.RECIPROCAL_LUT,
    )

    payload = json.loads(write_variant_profile(tmp_path, config).read_text(encoding="utf-8"))

    assert payload["operation_counts"]["bdcn_materialized_probability_entries"] == 0
    assert payload["assumptions"]["hardware_measurement"] is False
