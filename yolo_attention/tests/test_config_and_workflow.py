from __future__ import annotations

from pathlib import Path

import pytest

from yolo_attention.config import (
    BasisKind,
    BDCNCodebookKind,
    BDCNDenominator,
    BDCNProjection,
    BDCNSharing,
    BiasKind,
    NormalizationKind,
    ScaleMode,
    VariantConfig,
)
from yolo_attention.experiments import ExperimentRegistry, Stage
from yolo_attention.workflow import ResearchWorkflow


def test_variant_config_round_trips_yaml(tmp_path: Path) -> None:
    config = VariantConfig(
        name="H-SCR",
        basis=BasisKind.HADAMARD,
        bias=BiasKind.NONE,
        scale_mode=ScaleMode.DYNAMIC,
        normalization=NormalizationKind.EXACT,
        use_ste=True,
    )
    path = tmp_path / "variant.yaml"

    config.to_yaml(path)

    assert VariantConfig.from_yaml(path) == config


def test_invalid_fp_lut_combination_is_rejected() -> None:
    with pytest.raises(ValueError, match="FP"):
        VariantConfig(
            name="invalid",
            basis=BasisKind.FP,
            normalization=NormalizationKind.INTEGER_LUT,
        )


def test_bdcn_config_round_trips_and_validates_scope(tmp_path: Path) -> None:
    config = VariantConfig(
        name="D1-PHEAD",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.PER_HEAD,
        bdcn_projection=BDCNProjection.TWO_POT,
        bdcn_denominator=BDCNDenominator.RECIPROCAL_LUT,
    )
    path = config.to_yaml(tmp_path / "bdcn.yaml")
    assert VariantConfig.from_yaml(path) == config

    with pytest.raises(ValueError, match="only apply to BDCN"):
        VariantConfig(name="bad", normalization=NormalizationKind.EXACT, bdcn_levels=8)


def test_bdcn_distance_range_derives_uniform_bucket_step_and_round_trips(tmp_path: Path) -> None:
    config = VariantConfig(
        name="D0V2-U17",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.FIXED_EXP,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.FLOAT,
        bdcn_denominator=BDCNDenominator.EXACT,
        bdcn_levels=17,
        bdcn_distance_max=8.0,
    )

    assert config.resolved_bdcn_step == pytest.approx(0.5)
    assert VariantConfig.from_yaml(config.to_yaml(tmp_path / "range.yaml")) == config


def test_registry_encodes_the_approved_funnel() -> None:
    registry = ExperimentRegistry.default()

    assert [run.variant.name for run in registry.for_stage(Stage.SCREENING)] == [
        "I-SCR",
        "H-SCR",
        "T5-SCR",
    ]
    assert {run.variant.name for run in registry.for_stage(Stage.RECOVERY)} == {"W-DIR", "W-PROG"}
    assert registry.get("Q2").conditional is True
    assert registry.get("Q2").epochs == 5


def test_workflow_places_normalization_before_optional_quantization() -> None:
    workflow = ResearchWorkflow.default()
    payload = workflow.to_dict()

    assert payload["main"][2]["runs"] == ["I-SCR", "H-SCR", "T5-SCR"]
    assert payload["main"][4]["runs"] == [
        "N0-EXACT",
        "N0-LUT",
        "N0-PWL",
        "N0-SHIFT",
        "N0-HSIG",
        "N0-RELU",
        "N0-MK1",
        "N0-MK3",
        "N0-MK5",
    ]
    assert payload["main"][5]["epochs"] == 5
    assert payload["main"][-1]["runs"] == ["A-FINAL"]
    assert payload["optional"][0]["runs"] == ["Q0", "Q1-L3A", "Q2"]


def test_workflow_places_bdcn_denominator_branch_before_final() -> None:
    payload = ResearchWorkflow.default().to_dict()
    steps = {step["key"]: step for step in payload["main"]}
    order = [step["key"] for step in payload["main"]]

    assert steps["bdcn-reference"]["runs"] == ["D0-IDX"]
    assert steps["bdcn-learning"]["runs"] == [
        "D1-SHARED",
        "D1-PATTN",
        "D1-PHEAD",
        "D1-SEED1 (conditional)",
    ]
    assert steps["bdcn-learning"]["epochs"] == 10
    assert steps["bdcn-projection"]["runs"] == ["D2-FP", "D2-1P", "D2-2P"]
    assert steps["bdcn-denominator"]["runs"] == ["R0-DIV", "R1-RLUT", "R2-PSHIFT"]
    assert order.index("normalization-recovery") < order.index("bdcn-reference")
    assert order.index("bdcn-denominator") < order.index("final")
    assert steps["final"]["selection"] == "compare A0, N1 winner, and BDCN winner"
