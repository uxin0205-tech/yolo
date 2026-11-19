from __future__ import annotations

import pytest
import torch
from ultralytics.nn.modules.block import C2PSA
from ultralytics.nn.tasks import DetectionModel

from yolo_attention.config import (
    BasisKind,
    BDCNCodebookKind,
    BDCNDenominator,
    BDCNProjection,
    BDCNSharing,
    NormalizationKind,
    ScaleMode,
    VariantConfig,
)
from yolo_attention.integration import (
    YOLO26M_ATTENTION_PATHS,
    bdcn_table_assignment,
    convert_yolo26_model,
    freeze_for_stage,
)
from yolo_attention.pwl_validation import PWLModelDiagnosticsCollector


def test_yolo26m_yaml_converts_both_attention_sites_on_cpu() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)

    paths = convert_yolo26_model(model, VariantConfig(name="I-SCR", basis=BasisKind.IDENTITY))

    c2psa_count = sum(isinstance(module, C2PSA) for module in model.modules())
    assert c2psa_count > 0
    assert paths == list(YOLO26M_ATTENTION_PATHS)
    assert len(paths) == 2


def test_pwl_collector_observes_both_yolo26_attention_sites() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False).eval()
    config = VariantConfig(
        name="PWL-EXACT",
        basis=BasisKind.HADAMARD,
        bias="decomposed_2d",
        scale_mode=ScaleMode.POWER_OF_TWO,
        normalization=NormalizationKind.EXACT,
    )
    convert_yolo26_model(model, config)
    for path in YOLO26M_ATTENTION_PATHS:
        model.get_submodule(path).score.set_fixed_coefficients(torch.ones(4, 2))

    with PWLModelDiagnosticsCollector(model) as collector, torch.no_grad():
        model(torch.randn(1, 3, 64, 64))

    summaries = collector.summaries()
    assert [summary["site"] for summary in summaries] == list(YOLO26M_ATTENTION_PATHS)
    assert all(summary["aggregate"]["count"] > 0 for summary in summaries)


def test_yolo26_conversion_fails_closed_when_expected_site_is_missing() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    model.model[22].m[0][1].attn = torch.nn.Identity()

    with pytest.raises(ValueError, match="expected Attention paths"):
        convert_yolo26_model(model, VariantConfig(name="I-SCR", basis=BasisKind.IDENTITY))


def test_yolo26_conversion_reconfigures_existing_hardware_attention() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    dynamic = VariantConfig(
        name="V1-DYN",
        basis=BasisKind.HADAMARD,
        scale_mode=ScaleMode.DYNAMIC,
    )
    power_of_two = VariantConfig(
        name="V1-P2",
        basis=BasisKind.HADAMARD,
        scale_mode=ScaleMode.POWER_OF_TWO,
    )
    convert_yolo26_model(model, dynamic)
    before = {
        path: model.get_submodule(path).qkv.q.conv.weight.detach().clone() for path in YOLO26M_ATTENTION_PATHS
    }

    paths = convert_yolo26_model(model, power_of_two)

    assert paths == list(YOLO26M_ATTENTION_PATHS)
    for path in YOLO26M_ATTENTION_PATHS:
        attention = model.get_submodule(path)
        assert attention.config == power_of_two
        assert attention.score.scale_mode is ScaleMode.POWER_OF_TWO
        torch.testing.assert_close(attention.qkv.q.conv.weight, before[path])


@pytest.mark.parametrize(
    ("sharing", "expected"),
    [
        (BDCNSharing.GLOBAL, ([0, 0, 0, 0], [0, 0, 0, 0], 1)),
        (BDCNSharing.PER_ATTENTION, ([0, 0, 0, 0], [1, 1, 1, 1], 2)),
        (BDCNSharing.PER_HEAD, ([0, 1, 2, 3], [4, 5, 6, 7], 8)),
    ],
)
def test_bdcn_table_assignment(sharing: BDCNSharing, expected: tuple[list[int], list[int], int]) -> None:
    indices, tables = bdcn_table_assignment(sharing, sites=2, heads=4)
    assert ([*indices[0]], [*indices[1]], tables) == expected


def test_bdcn_global_bank_is_shared_by_both_attention_sites() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    config = VariantConfig(
        name="D1-SHARED",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.FLOAT,
        bdcn_denominator=BDCNDenominator.EXACT,
    )

    convert_yolo26_model(model, config)
    first = model.get_submodule(YOLO26M_ATTENTION_PATHS[0]).normalize
    second = model.get_submodule(YOLO26M_ATTENTION_PATHS[1]).normalize

    assert first.bank is second.bank
    assert first.bank.raw_ratios.data_ptr() == second.bank.raw_ratios.data_ptr()


def test_bdcn_reconfiguration_preserves_trained_shared_codebook() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    learned = VariantConfig(
        name="D1-SHARED",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.FLOAT,
        bdcn_denominator=BDCNDenominator.EXACT,
    )
    projected = VariantConfig(
        name="D2-1P",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.ONE_POT,
        bdcn_denominator=BDCNDenominator.EXACT,
    )
    convert_yolo26_model(model, learned)
    source_bank = model.get_submodule(YOLO26M_ATTENTION_PATHS[0]).normalize.bank
    with torch.no_grad():
        source_bank.raw_ratios.copy_(
            torch.linspace(-0.75, 0.75, source_bank.raw_ratios.numel()).view_as(source_bank.raw_ratios)
        )
    expected = source_bank.raw_ratios.detach().clone()

    convert_yolo26_model(model, projected)

    first = model.get_submodule(YOLO26M_ATTENTION_PATHS[0]).normalize.bank
    second = model.get_submodule(YOLO26M_ATTENTION_PATHS[1]).normalize.bank
    torch.testing.assert_close(first.raw_ratios, expected)
    assert first is second


def test_bdcn_codebook_stage_only_trains_shared_codebook() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    config = VariantConfig(
        name="D1-SHARED",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.FLOAT,
        bdcn_denominator=BDCNDenominator.EXACT,
    )
    convert_yolo26_model(model, config)

    summary = freeze_for_stage(model, "bdcn_codebook")

    assert summary.trainable_parameters == 15
    assert len(summary.trainable_names) == 1
    assert summary.trainable_names[0].endswith("normalize.bank.raw_ratios")


def test_bdcn_fused_value_forwards_both_yolo26_attention_sites() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False).eval()
    config = VariantConfig(
        name="R1-RLUT",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.TWO_POT,
        bdcn_denominator=BDCNDenominator.RECIPROCAL_LUT,
    )
    convert_yolo26_model(model, config)

    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))

    assert isinstance(output, tuple)
    for path in YOLO26M_ATTENTION_PATHS:
        attention = model.get_submodule(path)
        assert attention.last_scores is not None
        assert attention.last_probabilities is None
