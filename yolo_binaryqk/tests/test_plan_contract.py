from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from ultralytics.nn.modules.block import Attention

from binary_attention.attention.base import (
    MagnitudeSideChannelAttention,
    ParallelDualBinaryAttention,
    ResidualDualFullBasisAttention,
    ResidualDualMatchedBasisAttention,
)
from binary_attention.attention.quantizers import fake_quant_v
from binary_attention.kd import add_kd_loss, prediction_distillation_loss
from binary_attention.ema_repair import latest_repairable
from binary_attention.trainer import _frozen_teacher_prediction, _validation_counter_model, freeze_to_attention_only
from binary_attention.training_profiles import FORMAL_TRAINING_ARGS, make_training_overrides
from binary_attention.variants.definitions import (
    VARIANTS,
    get_variant,
    materialize_inherited_variant,
    materialize_non_kd_bias_variant,
    materialize_t6_candidate,
    materialize_t7_variant,
    materialize_selected_t3,
    quantization_contract,
    variant_from_resolved_config,
)
from binary_attention.workflow import create_run_artifact
from binary_attention.report import _formal_runs, build_summary


def test_attention_only_formal_profile_is_ten_epochs():
    assert FORMAL_TRAINING_ARGS["epochs"] == 10
    assert FORMAL_TRAINING_ARGS["batch"] == 128
    assert FORMAL_TRAINING_ARGS["seed"] == 0
    assert FORMAL_TRAINING_ARGS["deterministic"] is True
    assert FORMAL_TRAINING_ARGS["optimizer"] == "AdamW"
    assert FORMAL_TRAINING_ARGS["lr0"] == 5e-5
    assert FORMAL_TRAINING_ARGS["lrf"] == 0.1
    assert FORMAL_TRAINING_ARGS["weight_decay"] == 0.02
    t7_pv = materialize_t7_variant(
        "T7-PV", base_variant="T3", kd_components=("feature",), bias_type="dense_2d"
    )
    assert t7_pv.use_distillation is True
    assert t7_pv.bias_type == "dense_2d"
    assert t7_pv.p_bits == 8 and t7_pv.v_bits == 8
    full = make_training_overrides(
        stage="full", model="model.yaml", data="data.yaml", device="cpu", project="runs", name="y"
    )
    assert full["epochs"] == 10
    with pytest.raises(ValueError):
        make_training_overrides(
            stage="pilot", model="model.yaml", data="data.yaml", device="cpu", project="runs", name="x"
        )


def test_only_attention_parameters_remain_trainable():
    model = nn.Module()
    model.stem = nn.Conv2d(16, 16, 1)
    model.attn = Attention(16, num_heads=1)
    names = freeze_to_attention_only(model)
    assert names and all(name.startswith("attn.") for name in names)
    assert all(not parameter.requires_grad for parameter in model.stem.parameters())
    assert all(parameter.requires_grad for parameter in model.attn.parameters())


def test_e2_dual_is_zero_training_matched_dual_basis():
    variant = get_variant("E2-DUAL")
    assert variant.use_qat is False
    assert variant.qk_mode == "dual"
    assert variant.attention_type == "ResidualDualMatchedBasisAttention"
    assert variant.theoretical_cost == {"binary_qk": 2, "softmax": 1, "pv": 1}


def test_scaled_binary_attention_uses_one_scale_per_sample_and_head():
    from binary_attention.attention.base import ScaledBinaryAttention

    module = ScaledBinaryAttention(16, num_heads=2)
    q = torch.arange(1, 1 + 2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)
    k = -q.roll(1, dims=-1)
    quantized_q, quantized_k = module.binary_qk(q, k)
    expected_q_scale = q.abs().mean(dim=(2, 3), keepdim=True)
    expected_k_scale = k.abs().mean(dim=(2, 3), keepdim=True)
    assert torch.equal(quantized_q.abs(), expected_q_scale.expand_as(q))
    assert torch.equal(quantized_k.abs(), expected_k_scale.expand_as(k))


def test_v8_uses_independent_channel_scales_across_tokens():
    value = torch.tensor([[[[1.0, -2.0, 0.5], [10.0, -5.0, 0.0]]]])
    quantized = fake_quant_v(value, 8)
    scales = value.abs().amax(dim=-1, keepdim=True) / 127
    expected = scales * torch.round((value / scales).clamp(-127, 127))
    assert torch.equal(quantized, expected)
    assert not torch.equal(scales[..., 0, :], scales[..., 1, :])


def test_paper_main_quantization_contract_is_explicit():
    assert quantization_contract(materialize_t7_variant(
        "T7-PV", base_variant="T2", kd_components=("feature",), bias_type="dense_2d"
    )) == {
        "qk_scale": "mean_abs_channel_token_per_sample_head",
        "p8_scale": "static_unsigned_1_over_255",
        "v8_scale": "max_abs_token_per_sample_head_channel",
        "bias_parameterization": "full_2d_relative_position",
        "bias_initialization": "truncated_normal_std_0.02",
    }


def test_n_series_forward_costs_and_cross_terms():
    x = torch.randn(1, 16, 4, 4)
    cases = [
        (ParallelDualBinaryAttention(16, num_heads=2), (2, 2, 2)),
        (ResidualDualFullBasisAttention(16, num_heads=2), (4, 1, 1)),
        (ResidualDualMatchedBasisAttention(16, num_heads=2), (2, 1, 1)),
    ]
    for module, expected in cases:
        module(x)
        actual = (int(module.binary_qk_count), int(module.softmax_count), int(module.pv_count))
        assert actual == expected

    n2 = ResidualDualFullBasisAttention(16, num_heads=2)
    n3 = ResidualDualMatchedBasisAttention(16, num_heads=2)
    n3.load_state_dict(n2.state_dict())
    with torch.no_grad():
        n2.basis_lambda[0, 1] = 0
        n2.basis_lambda[1, 0] = 0
    n2(x)
    n3(x)
    assert torch.allclose(n2.last_scores, n3.last_scores, atol=1e-6, rtol=1e-5)

    n3 = ResidualDualMatchedBasisAttention(16, num_heads=2)
    with patch.object(n3, "_dot", wraps=n3._dot) as dot:
        n3(x)
    assert dot.call_count == 2


def test_n4_magnitude_changes_score_and_quantized_modes_are_real():
    x = torch.randn(1, 16, 4, 4)
    module = MagnitudeSideChannelAttention(16, num_heads=2)
    with torch.no_grad():
        module.magnitude_weight.zero_()
    module(x)
    without_magnitude = module.last_scores.clone()
    with torch.no_grad():
        module.magnitude_weight.fill_(1)
    module(x)
    assert not torch.allclose(without_magnitude, module.last_scores)
    assert get_variant("N4-I8").magnitude_bits == 8
    assert get_variant("N4-I4").magnitude_bits == 4
    assert get_variant("N4-PV").p_bits == 8 and get_variant("N4-PV").v_bits == 8


def test_post_t3_variants_inherit_selected_configuration():
    components = ("positional", "attention")
    t7 = materialize_t7_variant(
        "T7-D", base_variant="T3", kd_components=components
    )
    assert t7.kd_components == components and t7.bias_type == "dense_2d"
    t7_p = materialize_t7_variant(
        "T7-P", base_variant="T3", kd_components=components, bias_type="decomposed_2d"
    )
    assert t7_p.use_distillation is True
    assert t7_p.kd_components == components and t7_p.bias_type == "decomposed_2d"
    n4 = materialize_non_kd_bias_variant(
        "N4-PV", bias_type="dense_2d", magnitude_mode="int4", parent_variant="T7-P"
    )
    assert n4.magnitude_bits == 4 and n4.p_bits == n4.v_bits == 8
    resolved = {**n4.__dict__, "config_hash": n4.config_hash}
    assert variant_from_resolved_config(resolved) == n4
    with pytest.raises(ValueError):
        materialize_non_kd_bias_variant("N4-FP", bias_type="invalid")


def test_t3_scope_and_n4_non_kd_contract():
    assert get_variant("T6-O").kd_target_family == "T1-T5"
    t6 = materialize_t6_candidate("T6", base_variant="T3", components=("positional", "attention"))
    assert t6.kd_target_family == "T1-T5"
    n4 = materialize_non_kd_bias_variant("N4-I8", bias_type="dense_2d", parent_variant="T7-D")
    assert n4.use_distillation is False
    assert n4.kd_components == ()
    assert n4.kd_target_family is None
    assert n4.magnitude_bits == 8


def test_dual_attention_exposes_real_differentiable_kd_probability():
    module = ResidualDualMatchedBasisAttention(16, num_heads=2, use_qat=True).train()
    output = module(torch.randn(1, 16, 4, 4, requires_grad=True))
    probability = module.kd_probabilities
    assert probability is not None and probability.shape == (1, 2, 16, 16)
    total, kd = add_kd_loss(
        output.square().mean(), output, output.detach(), ("attention",), 0.1,
        attention_pairs=[(probability, probability.detach().roll(1, dims=-1))],
    )
    total.backward()
    assert kd.item() > 0
    assert module.qkv.conv.weight.grad is not None


def test_kd_is_added_to_total_loss_and_teacher_stays_detached():
    student = torch.randn(2, 8, requires_grad=True)
    teacher = torch.randn(2, 8, requires_grad=True)
    detection = student.square().mean()
    total, kd = add_kd_loss(detection, student, teacher, ("output",), 0.1)
    total.backward()
    assert kd.item() > 0
    assert teacher.grad is None
    assert student.grad is not None


def test_hard_detection_kd_uses_teacher_pseudo_labels():
    student = {
        "boxes": torch.randn(2, 64, 5, requires_grad=True),
        "scores": torch.randn(2, 3, 5, requires_grad=True),
    }
    teacher = {
        "boxes": torch.randn(2, 64, 5, requires_grad=True),
        "scores": torch.randn(2, 3, 5, requires_grad=True),
    }
    total, kd = add_kd_loss(student["boxes"].sum() * 0, student, teacher, ("hard",), 0.1)
    total.backward()
    assert kd.item() > 0
    assert all(value.grad is None for value in teacher.values())
    assert all(value.grad is not None for value in student.values())


def test_output_kd_excludes_detect_feature_maps():
    student = {
        "boxes": torch.randn(2, 64, 5, requires_grad=True),
        "scores": torch.randn(2, 3, 5, requires_grad=True),
        "feats": [torch.full((2, 4, 2, 2), 1000.0, requires_grad=True)],
    }
    teacher = {
        "boxes": torch.randn(2, 64, 5),
        "scores": torch.randn(2, 3, 5),
        "feats": [torch.full((2, 4, 2, 2), -1000.0)],
    }
    loss = prediction_distillation_loss(student, teacher, ("output",))
    expected = torch.stack([
        torch.nn.functional.mse_loss(student["boxes"], teacher["boxes"]),
        torch.nn.functional.mse_loss(student["scores"], teacher["scores"]),
    ]).mean()
    assert torch.allclose(loss, expected)


def test_frozen_teacher_raw_prediction_keeps_child_bn_in_eval():
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(2)

        def forward(self, x):
            return {"training": self.training, "value": self.bn(x)}

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([Head()])

        def forward(self, x):
            return self.model[-1](x)

    teacher = Teacher().eval()
    before = teacher.model[-1].bn.running_mean.clone()
    prediction = _frozen_teacher_prediction(teacher, torch.randn(2, 2, 3, 3))
    assert prediction["training"] is True
    assert teacher.model[-1].training is False
    assert teacher.model[-1].bn.training is False
    assert torch.equal(before, teacher.model[-1].bn.running_mean)


def test_validation_counter_tracks_the_model_used_by_ultralytics():
    class State:
        pass

    trainer = State()
    trainer.model = nn.Linear(2, 2)
    trainer.ema = State()
    trainer.ema.ema = nn.Linear(2, 2)
    assert _validation_counter_model(trainer) is trainer.ema.ema
    trainer.ema.ema = None
    assert _validation_counter_model(trainer) is trainer.model


def test_latest_ema_repair_candidate_requires_completed_formal_run(tmp_path: Path):
    run = tmp_path / "artifacts/runs/T0/full/run-id"
    (run / "ultralytics/train/weights").mkdir(parents=True)
    (run / "ultralytics/train/weights/last.pt").touch()
    (run / "status.json").write_text(json.dumps({"completed": True, "valid_for_research": True}))
    (run / "resolved_config.json").write_text(json.dumps({
        "id": "T0",
        "plan_name": "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan",
    }))
    assert latest_repairable(tmp_path, "T0") == run
    status = json.loads((run / "status.json").read_text())
    status["checkpoint_weight_source"] = "epoch_ema"
    (run / "status.json").write_text(json.dumps(status))
    assert latest_repairable(tmp_path, "T0") is None


def test_attention_kd_requires_real_attention_maps_and_backpropagates():
    prediction = torch.randn(1, 3, requires_grad=True)
    teacher_prediction = torch.randn(1, 3)
    with pytest.raises(ValueError, match="attention maps"):
        add_kd_loss(prediction.square().mean(), prediction, teacher_prediction, ("attention",), 0.1)

    student_logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    teacher_logits = torch.randn(1, 2, 4, 4)
    student_attention = student_logits.softmax(-1)
    teacher_attention = teacher_logits.softmax(-1)
    total, kd = add_kd_loss(
        prediction.square().mean(), prediction, teacher_prediction, ("attention",), 0.1,
        attention_pairs=[(student_attention, teacher_attention)],
    )
    total.backward()
    assert kd.item() > 0
    assert student_logits.grad is not None


def test_run_artifact_has_required_reconstruction_files(tmp_path: Path):
    run = create_run_artifact(tmp_path, get_variant("T2"), "full", ["test"])
    required = {
        "experiment_report.md", "resolved_config.json", "training_args.json",
        "architecture_manifest.json", "checkpoint_manifest.json", "validation_metrics.json",
        "attention_diagnostics.json", "training_curves.csv", "logs", "checkpoints",
        "parameter_delta_diagnostics.json",
    }
    assert required.issubset({path.name for path in run.iterdir()})
    config = (run / "resolved_config.json").read_text()
    assert "not executed by plan" in config
    status = json.loads((run / "status.json").read_text())
    args = json.loads((run / "training_args.json").read_text())
    assert status["stage"] == "full"
    assert status["epochs"] == args["epochs"] == 10


def test_runner_reuse_is_bound_to_current_resolved_config_hash():
    runner = (Path(__file__).parents[1] / "run_formal_plan.sh").read_text()
    assert 'resolved.get("config_hash") == expected_config_hash' in runner
    assert "expected_config_hash()" in runner
    assert "contract_ok" in runner
    assert "max(values, key=lambda item: (item[0], item[1]))[0]" in runner


def test_final_report_discloses_adaptation_and_quantization_contract(tmp_path: Path):
    build_summary(tmp_path)
    text = (tmp_path / "artifacts/reports/binary_attention_final_report.md").read_text()
    for expected in (
        "10-epoch attention-only adaptation",
        "300-epoch full-model",
        "E2-DUAL",
        "mean_abs_channel_token_per_sample_head",
        "epoch-last EMA",
        "fake quantization",
    ):
        assert expected in text


def test_report_selects_one_canonical_expected_stage_per_variant(tmp_path: Path):
    def make_run(variant: str, stage: str, run_id: str, plan: str) -> Path:
        run = tmp_path / "artifacts/runs" / variant.replace("/", "-") / stage / run_id
        run.mkdir(parents=True)
        (run / "resolved_config.json").write_text(json.dumps({"id": variant, "plan_name": plan}))
        (run / "status.json").write_text(json.dumps({
            "stage": stage, "completed": True, "valid_for_research": True,
        }))
        return run

    plan = "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan"
    make_run("E0", "validation", "older", plan)
    latest = make_run("E0", "validation", "latest", plan)
    make_run("N4-PV", "validation", "legacy-draft", "YOLO11 BinaryAttention simplified full-COCO plan")
    make_run("T0", "full", "formal", plan)
    selected = _formal_runs(tmp_path)
    assert latest in selected
    assert len([run for run in selected if run.parent.parent.name == "E0"]) == 1
    assert all(run.parent.name != "validation" or run.parent.parent.name in {"E0", "E1-S", "E1", "E2-DUAL"} for run in selected)
