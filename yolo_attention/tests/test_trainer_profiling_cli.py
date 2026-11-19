from __future__ import annotations

import json

import pytest
from ultralytics.models.yolo.detect.train import DetectionTrainer

from yolo_attention.cli import main
from yolo_attention.config import BasisKind, VariantConfig
from yolo_attention.profiling import AttentionShape, estimate_operations, write_variant_profile
from yolo_attention.training import build_hardware_attention_model, fail_on_nonfinite_loss, make_trainer


def test_trainer_factory_carries_variant_and_scope_without_starting_training() -> None:
    variant = VariantConfig(name="H-SCR", basis=BasisKind.HADAMARD)

    factory = make_trainer(variant, stage="screening")

    assert issubclass(factory.trainer_class, DetectionTrainer)
    assert factory.variant_config == variant
    assert factory.training_stage == "screening"


def test_training_loss_guard_rejects_nonfinite_batch_immediately() -> None:
    import torch

    trainer = type("TrainerState", (), {"loss": torch.tensor(float("nan"))})()

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        fail_on_nonfinite_loss(trainer)


def test_operation_estimate_matches_plan_reference_shape() -> None:
    report = estimate_operations(AttentionShape(tokens=400, heads=4, key_dim=32, value_dim=64))

    assert report.fp_qk_mac == 20_480_000
    assert report.pv_mac == 40_960_000
    assert report.dual_binary_word_ops == 1_280_000
    assert report.hadamard_add_sub == 512_000
    assert report.bdcn_score_lookups == 640_000
    assert report.bdcn_value_bucket_add == 40_960_000
    assert report.bdcn_codebook_value_ops == 1_638_400
    assert report.bdcn_reciprocal_rows == 1_600
    assert report.bdcn_materialized_probability_entries == 0


def test_hadamard_variant_profile_counts_both_binary_bases(tmp_path) -> None:
    destination = write_variant_profile(
        tmp_path,
        VariantConfig(name="H-EXACT", basis=BasisKind.HADAMARD),
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["selected_operation_counts"]["binary_word_ops"] == 2_560_000
    assert payload["arithmetic_cost_proxy"] == 91_910_400


def test_cli_lists_registry_as_json(capsys) -> None:
    exit_code = main(["list", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["screening"] == ["I-SCR", "H-SCR", "T5-SCR"]
    assert payload["conditional"] == ["V1-SHEAD", "V1-P2", "Q2"]


def test_cli_prints_complete_workflow_without_running_training(capsys) -> None:
    exit_code = main(["workflow", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["main"][-1]["runs"] == ["A-FINAL"]
    assert payload["optional"][0]["optional"] is True


def test_hardware_trainer_builds_custom_structure_before_loading_parent_state() -> None:
    import torch
    from ultralytics.nn.tasks import DetectionModel

    from yolo_attention.config import ScaleMode
    from yolo_attention.integration import YOLO26M_ATTENTION_PATHS, convert_yolo26_model

    variant = VariantConfig(
        name="V1-P2",
        basis=BasisKind.HADAMARD,
        scale_mode=ScaleMode.POWER_OF_TWO,
    )
    source = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    convert_yolo26_model(source, variant)
    for path in YOLO26M_ATTENTION_PATHS:
        attention = source.get_submodule(path)
        attention.score.gamma.data.fill_(0.375)
        attention.score.set_fixed_coefficients(torch.full((attention.num_heads, 2), 0.25))

    restored = build_hardware_attention_model(
        cfg="yolo26m.yaml",
        nc=80,
        channels=3,
        weights=source,
        variant=variant,
        verbose=False,
    )

    for path in YOLO26M_ATTENTION_PATHS:
        score = restored.get_submodule(path).score
        torch.testing.assert_close(score.gamma, torch.full_like(score.gamma, 0.375))
        torch.testing.assert_close(score.fixed_coefficients, torch.full_like(score.fixed_coefficients, 0.25))
        assert bool(score.fixed_coefficients_ready)


def test_hardware_trainer_rejects_fused_custom_parent_with_incomplete_state() -> None:
    from ultralytics.nn.tasks import DetectionModel

    from yolo_attention.integration import convert_yolo26_model

    variant = VariantConfig(name="V1-B0", basis=BasisKind.HADAMARD)
    source = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    convert_yolo26_model(source, variant)
    source.fuse(verbose=False)

    with pytest.raises(RuntimeError, match="incompatible custom parent checkpoint"):
        build_hardware_attention_model(
            cfg="yolo26m.yaml",
            nc=80,
            channels=3,
            weights=source,
            variant=variant,
            verbose=False,
        )
