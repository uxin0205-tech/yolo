from __future__ import annotations

import copy
import itertools
from typing import cast

import pytest
import torch
from torch import nn

from yolo_quantize import (
    Full35WeightViews,
    WeightAnalysisPlan,
    WeightFormatAnalyzer,
    WeightViewParityManifest,
)


class _TinyCatalogModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph = nn.Module()
        self.graph.model = nn.ModuleList([nn.Module()])
        self.graph.model[0].conv = nn.Conv2d(1, 1, (1, 5), bias=False)


def _tiny_views() -> Full35WeightViews:
    master = _TinyCatalogModel()
    with torch.no_grad():
        master.graph.model[0].conv.weight.copy_(
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).reshape(1, 1, 1, 5)
        )
    return Full35WeightViews(
        master=master,
        deployment=copy.deepcopy(master),
        manifest=cast(WeightViewParityManifest, None),
    )


def _group_views() -> Full35WeightViews:
    master = _TinyCatalogModel()
    master.graph.model[0].conv = nn.Conv2d(1, 2, (1, 65), bias=False)
    with torch.no_grad():
        master.graph.model[0].conv.weight.copy_(
            torch.linspace(-1.0, 1.0, 130).reshape(2, 1, 1, 65)
        )
    return Full35WeightViews(
        master=master,
        deployment=copy.deepcopy(master),
        manifest=cast(WeightViewParityManifest, None),
    )


def test_analyzer_can_profile_one_locked_deployment_model_directly() -> None:
    model = _tiny_views().deployment
    plan = WeightAnalysisPlan(
        view_names=("deployment",),
        regions=("backbone_early",),
        uniform_bits=(4,),
        uniform_granularities=("per_output_channel",),
        uniform_scale_methods=("optimal_scaled_codebook",),
        include_fixed_sd4=False,
        include_paper_twn=False,
    )

    analysis = WeightFormatAnalyzer().analyze_deployment(model, plan)

    assert tuple(analysis.view_catalogs) == ("deployment",)
    assert len(analysis.measurements) == 1
    assert analysis.measurements[0].path == "graph.model.0.conv"


def test_analyzer_includes_every_uniform_bit_from_w8_through_w4() -> None:
    analysis = WeightFormatAnalyzer().analyze(
        _tiny_views(),
        WeightAnalysisPlan(
            view_names=("deployment",),
            regions=("backbone_early",),
            uniform_bits=(8, 7, 6, 5, 4),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("max",),
            include_fixed_sd4=False,
            include_paper_twn=False,
        ),
    )

    assert tuple(
        (
            item.format_id,
            item.bits,
            item.elements,
            item.scale_count,
            item.code_bytes,
        )
        for item in analysis.measurements
    ) == (
        ("uniform-w8-per_output_channel-max", 8, 5, 1, 5),
        ("uniform-w7-per_output_channel-max", 7, 5, 1, 5),
        ("uniform-w6-per_output_channel-max", 6, 5, 1, 4),
        ("uniform-w5-per_output_channel-max", 5, 5, 1, 4),
        ("uniform-w4-per_output_channel-max", 4, 5, 1, 3),
    )


def test_analyzer_reports_scale_metadata_for_all_uniform_granularities() -> None:
    analysis = WeightFormatAnalyzer().analyze(
        _group_views(),
        WeightAnalysisPlan(
            view_names=("deployment",),
            regions=("backbone_early",),
            uniform_bits=(8,),
            uniform_granularities=(
                "per_tensor",
                "per_output_channel",
                "group32",
                "group64",
            ),
            uniform_scale_methods=("max",),
            include_fixed_sd4=False,
            include_paper_twn=False,
        ),
    )

    assert tuple(
        (item.granularity, item.scale_count, item.metadata_bytes)
        for item in analysis.measurements
    ) == (
        ("per_tensor", 1, 4),
        ("per_output_channel", 2, 8),
        ("group32", 6, 24),
        ("group64", 4, 16),
    )


def test_analyzer_reconstructs_fixed_sd4_with_its_hardware_codebook() -> None:
    analysis = WeightFormatAnalyzer().analyze(
        _tiny_views(),
        WeightAnalysisPlan(
            view_names=("deployment",),
            regions=("backbone_early",),
            uniform_bits=(4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("max",),
            fixed_sd4_granularities=("per_output_channel",),
            fixed_sd4_scale_methods=("max",),
            include_fixed_sd4=True,
            include_paper_twn=False,
        ),
    )
    sd4 = next(item for item in analysis.measurements if item.family == "fixed_sd4")

    assert {
        "format_id": sd4.format_id,
        "bits": sd4.bits,
        "scale_count": sd4.scale_count,
        "code_bytes": sd4.code_bytes,
        "metadata_bytes": sd4.metadata_bytes,
        "mse": sd4.numeric["mse"],
        "occupied_codes": sd4.numeric["occupied_codes"],
        "code_min": sd4.numeric["code_min"],
        "code_max": sd4.numeric["code_max"],
    } == {
        "format_id": "fixed-sd4-per_output_channel-max",
        "bits": 4,
        "scale_count": 1,
        "code_bytes": 3,
        "metadata_bytes": 4,
        "mse": 0.0,
        "occupied_codes": 5,
        "code_min": -7,
        "code_max": 7,
    }


def test_analyzer_applies_the_paper_twn_threshold_and_shared_alpha() -> None:
    analysis = WeightFormatAnalyzer().analyze(
        _tiny_views(),
        WeightAnalysisPlan(
            view_names=("deployment",),
            regions=("backbone_early",),
            uniform_bits=(4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("max",),
            include_fixed_sd4=False,
            include_paper_twn=True,
        ),
    )
    ternary = next(item for item in analysis.measurements if item.family == "paper_twn")

    assert {
        "format_id": ternary.format_id,
        "bits": ternary.bits,
        "scale_count": ternary.scale_count,
        "code_bytes": ternary.code_bytes,
        "metadata_bytes": ternary.metadata_bytes,
        "threshold": round(float(ternary.numeric["threshold"]), 6),
        "alpha": round(float(ternary.numeric["alpha"]), 6),
        "mse": round(float(ternary.numeric["mse"]), 6),
        "zero_ratio": round(float(ternary.numeric["zero_ratio"]), 6),
        "occupied_codes": ternary.numeric["occupied_codes"],
    } == {
        "format_id": "paper-twn-layerwise",
        "bits": 2,
        "scale_count": 1,
        "code_bytes": 2,
        "metadata_bytes": 4,
        "threshold": 0.42,
        "alpha": 0.75,
        "mse": 0.05,
        "zero_ratio": 0.2,
        "occupied_codes": 3,
    }


def test_analyzer_keeps_twn_v3_and_exact_ternary_as_distinct_cpu_baselines() -> None:
    analysis = WeightFormatAnalyzer().analyze(
        _group_views(),
        WeightAnalysisPlan(
            view_names=("deployment",),
            regions=("backbone_early",),
            uniform_bits=(4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("max",),
            include_fixed_sd4=False,
            include_paper_twn=False,
            include_filterwise_twn=True,
            include_exact_scaled_ternary=True,
        ),
    )
    baselines = {item.family: item for item in analysis.measurements}

    filterwise = baselines["twn_filterwise"]
    exact = baselines["exact_scaled_ternary"]
    assert (
        filterwise.format_id,
        filterwise.granularity,
        filterwise.scale_count,
        filterwise.code_bytes,
        filterwise.metadata_bytes,
    ) == ("twn-v3-0.75-filterwise", "per_output_filter", 2, 33, 8)
    assert (
        exact.format_id,
        exact.granularity,
        exact.scale_count,
        exact.code_bytes,
        exact.metadata_bytes,
    ) == ("exact-scaled-ternary-per_tensor", "per_tensor", 1, 33, 4)
    assert exact.numeric["mse"] >= 0.0
    assert filterwise.numeric["mse"] >= 0.0


def test_analysis_plan_rejects_unknown_granularity_instead_of_aliasing_it() -> None:
    with pytest.raises(ValueError, match="unsupported weight granularity"):
        WeightAnalysisPlan(
            uniform_granularities=cast(
                tuple[object, ...],
                ("group16",),
            ),
        )


def test_analysis_plan_rejects_unknown_sd4_scale_method() -> None:
    with pytest.raises(ValueError, match="unsupported weight scale method"):
        WeightAnalysisPlan(
            fixed_sd4_scale_methods=cast(
                tuple[object, ...],
                ("median",),
            ),
        )


def test_optimal_sd4_scaled_codebook_is_never_worse_than_legacy_grid() -> None:
    views = _tiny_views()
    with torch.no_grad():
        views.deployment.graph.model[0].conv.weight.copy_(
            torch.tensor([-7.3, -1.7, -0.49, 0.13, 2.6]).reshape(1, 1, 1, 5)
        )
    analysis = WeightFormatAnalyzer().analyze(
        views,
        WeightAnalysisPlan(
            view_names=("deployment",),
            regions=("backbone_early",),
            uniform_bits=(4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("max",),
            fixed_sd4_granularities=("per_output_channel",),
            fixed_sd4_scale_methods=(
                "mse_grid_v1",
                "optimal_scaled_codebook",
            ),
            include_fixed_sd4=True,
            include_paper_twn=False,
        ),
    )
    sd4 = {
        item.scale_method: item
        for item in analysis.measurements
        if item.family == "fixed_sd4"
    }

    assert (
        sd4["optimal_scaled_codebook"].numeric["mse"]
        <= sd4["mse_grid_v1"].numeric["mse"]
    )
    assert sd4["optimal_scaled_codebook"].format_id == (
        "fixed-sd4-per_output_channel-optimal_scaled_codebook"
    )


def test_optimal_sd4_handles_an_all_zero_group_deterministically() -> None:
    views = _tiny_views()
    with torch.no_grad():
        views.deployment.graph.model[0].conv.weight.zero_()
    analysis = WeightFormatAnalyzer().analyze(
        views,
        WeightAnalysisPlan(
            view_names=("deployment",),
            regions=("backbone_early",),
            uniform_bits=(4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=("max",),
            fixed_sd4_granularities=("per_output_channel",),
            fixed_sd4_scale_methods=("optimal_scaled_codebook",),
            include_fixed_sd4=True,
            include_paper_twn=False,
        ),
    )
    sd4 = next(item for item in analysis.measurements if item.family == "fixed_sd4")

    assert sd4.numeric["mse"] == 0.0
    assert sd4.numeric["occupied_codes"] == 1
    assert sd4.numeric["code_min"] == 0
    assert sd4.numeric["code_max"] == 0


def test_optimal_scaled_codebook_matches_independent_assignment_search() -> None:
    from yolo_quantize.scaled_codebook import optimal_scaled_codebook_scales

    values = torch.tensor([[7.3, 1.7, 0.49]])
    codebook = torch.tensor(
        [
            -1.0,
            -0.5,
            -0.25,
            -0.125,
            -0.0625,
            -0.03125,
            -0.015625,
            0.0,
            0.015625,
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
        ]
    )
    scale = optimal_scaled_codebook_scales(values, codebook=codebook)[0, 0]
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5
    reconstructed = codebook[torch.bucketize(values / scale, midpoints)] * scale
    exact_error = float((reconstructed - values).square().sum().item())

    positive_levels = codebook[codebook >= 0]
    brute_error = float("inf")
    for assignment in itertools.product(
        positive_levels.tolist(),
        repeat=values.numel(),
    ):
        codes = torch.tensor(assignment)
        denominator = float(codes.square().sum().item())
        candidate_scale = (
            0.0
            if denominator == 0.0
            else float((values.flatten() * codes).sum().item()) / denominator
        )
        if candidate_scale == 0.0:
            candidate = torch.zeros_like(values)
        else:
            candidate = (
                codebook[torch.bucketize(values / candidate_scale, midpoints)]
                * candidate_scale
            )
        brute_error = min(
            brute_error,
            float((candidate - values).square().sum().item()),
        )

    assert exact_error == pytest.approx(brute_error, abs=1e-6)


def test_optimal_scaled_codebook_handles_asymmetric_twos_complement_codes() -> None:
    from yolo_quantize.scaled_codebook import optimal_scaled_codebook_scales

    values = torch.tensor([[-7.3, 2.6, 0.13]])
    codebook = torch.arange(-8, 8, dtype=torch.float32)
    scale = optimal_scaled_codebook_scales(values, codebook=codebook)[0, 0]
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5
    reconstructed = codebook[torch.bucketize(values / scale, midpoints)] * scale
    exact_error = float((reconstructed - values).square().sum().item())

    brute_error = float("inf")
    for assignment in itertools.product(
        codebook.tolist(),
        repeat=values.numel(),
    ):
        codes = torch.tensor(assignment)
        denominator = float(codes.square().sum().item())
        numerator = float((values.flatten() * codes).sum().item())
        if denominator == 0.0 or numerator <= 0.0:
            continue
        candidate_scale = numerator / denominator
        candidate = (
            codebook[torch.bucketize(values / candidate_scale, midpoints)]
            * candidate_scale
        )
        brute_error = min(
            brute_error,
            float((candidate - values).square().sum().item()),
        )

    assert exact_error == pytest.approx(brute_error, abs=1e-6)


def test_exact_uniform_w4_is_never_worse_than_legacy_grid() -> None:
    views = _tiny_views()
    with torch.no_grad():
        views.deployment.graph.model[0].conv.weight.copy_(
            torch.tensor([-7.3, -1.7, -0.49, 0.13, 2.6]).reshape(1, 1, 1, 5)
        )
    analysis = WeightFormatAnalyzer().analyze(
        views,
        WeightAnalysisPlan(
            view_names=("deployment",),
            regions=("backbone_early",),
            uniform_bits=(4,),
            uniform_granularities=("per_output_channel",),
            uniform_scale_methods=(
                "mse_grid_v1",
                "optimal_scaled_codebook",
            ),
            include_fixed_sd4=False,
            include_paper_twn=False,
        ),
    )
    uniform = {item.scale_method: item for item in analysis.measurements}

    assert (
        uniform["optimal_scaled_codebook"].numeric["mse"]
        <= uniform["mse_grid_v1"].numeric["mse"]
    )
