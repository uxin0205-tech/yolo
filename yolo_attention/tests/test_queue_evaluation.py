from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_attention.config import BasisKind, ScaleMode, VariantConfig
from yolo_attention.evaluation import (
    BDCNDiagnosticsCollector,
    EvaluationRecipe,
    EvaluationRequest,
    ResultContractError,
    UltralyticsEvaluationBackend,
    collect_row_sum_max_error,
    resolve_run_outputs,
    standardize_metrics,
    write_standard_result,
)


def test_bdcn_diagnostics_collector_aggregates_forward_calls() -> None:
    import torch

    from yolo_attention.bdcn import BDCNCodebookBank, BDCNNormalizer
    from yolo_attention.config import BDCNCodebookKind, BDCNDenominator, BDCNProjection

    bank = BDCNCodebookBank(1, 17, 0.5, BDCNCodebookKind.FIXED_EXP, BDCNProjection.FLOAT)
    module = BDCNNormalizer(bank, torch.tensor([0]), 0.5, BDCNDenominator.EXACT)
    with BDCNDiagnosticsCollector(module) as collector:
        module(torch.tensor([[[[0.0, -2.0, -8.0, -10.0]]]]))
        module.aggregate(
            torch.tensor([[[[0.0, -1.0]]]]),
            torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        )

    result = collector.result()
    assert result.bucket_histogram[0] == 2
    assert result.bucket_histogram[-1] == 2
    assert result.bucket_overflow_rate == pytest.approx(1 / 6)
    assert result.distance_max == pytest.approx(10.0)
from yolo_attention.integration import YOLO26M_ATTENTION_PATHS


class FakeBox:
    map = 0.401
    map50 = 0.590
    map75 = 0.430
    maps = (0.1, 0.2, 0.3)


class FakeMetrics:
    box = FakeBox()


def test_evaluation_recipe_parses_without_training_fields(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "data: data/coco2017.yaml\nimgsz: 640\nbatch: 16\ndevice: '0'\n"
        "workers: 8\nsplit: val\nplots: false\n",
        encoding="utf-8",
    )

    recipe = EvaluationRecipe.from_yaml(path)

    assert recipe.to_ultralytics_args() == {
        "data": "data/coco2017.yaml",
        "imgsz": 640,
        "batch": 16,
        "device": "0",
        "workers": 8,
        "split": "val",
        "plots": False,
    }


def test_standardize_metrics_uses_explicit_coco_keys() -> None:
    metrics = standardize_metrics(FakeMetrics())

    assert metrics == {
        "map50_95": 0.401,
        "map50": 0.590,
        "map75": 0.430,
        "maps": [0.1, 0.2, 0.3],
    }


def test_standardize_metrics_rejects_missing_or_nonfinite_map() -> None:
    with pytest.raises(ResultContractError, match="box.map"):
        standardize_metrics(object())
    FakeBox.map = float("nan")
    try:
        with pytest.raises(ResultContractError, match="finite"):
            standardize_metrics(FakeMetrics())
    finally:
        FakeBox.map = 0.401


def test_collect_row_sum_error_supports_float_and_u8_diagnostics() -> None:
    import torch
    from torch import nn

    model = nn.Module()
    model.float_rows = nn.Module()
    model.float_rows.last_row_sums = torch.tensor([1.0, 1.003])
    model.u8_rows = nn.Module()
    model.u8_rows.last_row_sums = torch.tensor([254, 256], dtype=torch.int32)

    assert collect_row_sum_max_error(model) == pytest.approx(1 / 255)


def test_write_and_resolve_standard_result_requires_real_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint = run / "ultralytics" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    result = write_standard_result(
        run,
        standardize_metrics(FakeMetrics()),
        checkpoint_path=checkpoint,
        profile_path=None,
        row_sum_max_error=0.004,
    )
    restored = resolve_run_outputs(run, require_checkpoint=True)
    payload = json.loads((run / "metrics" / "queue-result.json").read_text(encoding="utf-8"))

    assert restored == result
    assert payload["map50_95"] == 0.401
    assert restored.checkpoint_path == str(checkpoint.resolve())
    checkpoint.unlink()
    with pytest.raises(ResultContractError, match="checkpoint"):
        resolve_run_outputs(run, require_checkpoint=True)


def test_official_backend_uses_recipe_and_parent_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "yolo26m.pt"
    checkpoint.touch()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights

        def val(self, **kwargs):
            calls.append((self.weights, kwargs))
            return FakeMetrics()

    recipe = EvaluationRecipe("data/coco2017.yaml", 640, 16, "0", 8, "val", False)
    request = EvaluationRequest(
        run_id="b26-fp",
        run_dir=tmp_path / "run",
        parent_checkpoint=checkpoint,
        recipe=recipe,
    )

    result = UltralyticsEvaluationBackend(model_factory=FakeYOLO).evaluate_official(request)

    assert result.map50_95 == 0.401
    assert calls[0][0] == str(checkpoint.resolve())
    assert calls[0][1]["data"] == "data/coco2017.yaml"


def test_variant_backend_converts_both_attention_sites_before_val(tmp_path: Path) -> None:
    from ultralytics.nn.tasks import DetectionModel

    from yolo_attention.attention import HardwareFriendlyAttention

    checkpoint = tmp_path / "parent.pt"
    checkpoint.touch()
    variant_path = tmp_path / "variant.yaml"
    VariantConfig(name="I-SCR", basis=BasisKind.IDENTITY).to_yaml(variant_path)

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)

        def val(self, **kwargs):
            assert all(
                isinstance(self.model.get_submodule(path), HardwareFriendlyAttention)
                for path in YOLO26M_ATTENTION_PATHS
            )
            return FakeMetrics()

    request = EvaluationRequest(
        run_id="n0-lut",
        run_dir=tmp_path / "run",
        parent_checkpoint=checkpoint,
        recipe=EvaluationRecipe("data/coco2017.yaml", 640, 16, "0", 8, "val", False),
        variant_path=variant_path,
    )

    result = UltralyticsEvaluationBackend(model_factory=FakeYOLO).evaluate_variant(request)

    assert result.map50_95 == 0.401


def test_fixed_scale_variant_calibrates_then_evaluates_and_exports_checkpoint(tmp_path: Path) -> None:
    import torch
    from ultralytics.nn.tasks import DetectionModel

    from yolo_attention.integration import convert_yolo26_model

    checkpoint = tmp_path / "parent.pt"
    checkpoint.touch()
    variant_path = tmp_path / "variant.yaml"
    VariantConfig(
        name="V1-P2",
        basis=BasisKind.HADAMARD,
        scale_mode=ScaleMode.POWER_OF_TWO,
    ).to_yaml(variant_path)
    calls: list[tuple[bool, bool]] = []

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False).eval()
            self.ckpt = {}
            convert_yolo26_model(
                self.model,
                VariantConfig(name="W-DIR", basis=BasisKind.HADAMARD),
            )

        def val(self, **kwargs):
            # Ultralytics validation fuses Conv+BN in place before inference.
            self.model.fuse(verbose=False)
            score = self.model.get_submodule(YOLO26M_ATTENTION_PATHS[0]).score
            calls.append((bool(score.calibration_enabled), bool(score.fixed_coefficients_ready)))
            with torch.inference_mode():
                self.model(torch.randn(1, 3, 64, 64))
                for path in YOLO26M_ATTENTION_PATHS:
                    score = self.model.get_submodule(path).score
                    score.fixed_coefficients = score.fixed_coefficients.clone()
            return FakeMetrics()

        def save(self, path: str) -> None:
            import copy

            torch.save({"model": copy.deepcopy(self.model).half()}, path)

    request = EvaluationRequest(
        run_id="v1-p2",
        run_dir=tmp_path / "run",
        parent_checkpoint=checkpoint,
        recipe=EvaluationRecipe("data/coco2017.yaml", 640, 16, "0", 8, "val", False),
        variant_path=variant_path,
    )

    result = UltralyticsEvaluationBackend(model_factory=FakeYOLO).evaluate_variant(request)

    assert calls == [(True, False), (False, True)]
    assert result.checkpoint_path == str((tmp_path / "run" / "checkpoints" / "calibrated.pt").resolve())
    assert Path(result.checkpoint_path).is_file()
    saved = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)["model"]
    saved_convs = [module for module in saved.modules() if hasattr(module, "conv")]
    assert any(hasattr(module, "bn") for module in saved_convs)
    for path in YOLO26M_ATTENTION_PATHS:
        assert bool(saved.get_submodule(path).score.fixed_coefficients_ready)
