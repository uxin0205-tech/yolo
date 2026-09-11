"""CPU 合約測試：不建立正式模型、不讀資料集、不使用 GPU。"""

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from torch import nn

from yolo_optimize.training import (
    HOGTaskLossRouter, SCOPE, TrainingModel, calibration_coefficient,
    capture_raw_p3, continuation_horizons, hog_multiplier,
    memory_headroom, severe_regressions,
    probe_gradient,
    ema_initial_updates, validation_boundary,
)
from yolo_combine.contracts import Task
from yolo_combine.joint_loss import TaskLossResult
from yolo_combine.joint_trainer import StageWarmupCosineScheduler
from yolo_combine.stage_policy import apply_stage, build_joint_optimizer
from yolo_combine.metrics import GATE_METRICS
from yolo_combine.resume import TrainingProgress, save_inference_weights, save_training_snapshot, load_training_snapshot
from ultralytics.utils.torch_utils import ModelEMA


def parent_payload():
    return {
        "resolved_config": {"stages": ["j0", "j1", "j2"], "enable_j3": False,
            "stage_policies": {"j0": {"task_mode": "pose", "epochs": 8},
                "j1": {"task_mode": "joint", "epochs": 20},
                "j2": {"task_mode": "joint", "epochs": 80},
                "j3": {"task_mode": "joint", "epochs": 20}}},
        "progress": {"stage": "j3"},
        "criteria_state": {
            "detect": {"end2end": True, "updates": 51, "o2m": .5, "o2o": .5,
                       "o2m_copy": .8, "final_o2m": .1, "total": 1.0},
            "pose": {"end2end": True, "updates": 59, "o2m": .4748031496062992,
                     "o2o": .5251968503937008, "o2m_copy": .8, "final_o2m": .1, "total": 1.0},
        },
    }


def test_parent_cli_j3_horizon_reconstructed_without_reset():
    assert continuation_horizons(parent_payload()) == {"detect": 120, "pose": 128}


def test_parent_horizon_drift_fails_closed():
    payload = parent_payload()
    payload["resolved_config"]["stage_policies"]["j3"]["epochs"] = 10
    with pytest.raises(ValueError, match="horizon"):
        continuation_horizons(payload)


def test_ema_age_mode_uses_only_paired_top_level_age():
    payload = {"ema_updates": 26597, "progress": {"global_macro_step": 123},
               "criteria_state": {"detect": {"updates": 51}}}
    assert ema_initial_updates("fresh", payload) == 0
    assert ema_initial_updates("parent", payload) == 26597
    with pytest.raises(ValueError, match="ema_age_mode"):
        ema_initial_updates("auto", payload)
    with pytest.raises(ValueError, match="頂層"):
        ema_initial_updates("parent", {"progress": {"global_macro_step": 26597}})
    with pytest.raises(ValueError, match="整數"):
        ema_initial_updates("parent", {"ema_updates": True})


def test_matching_scheduler_prefix_and_one_epoch_warmup():
    traces = []
    for budget in (5, 10):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1e-5, "group_name": "neck"}])
        scheduler = StageWarmupCosineScheduler(optimizer, stage="recovery", epochs=10,
            steps_per_epoch=3, warmup_epochs=1, warmup_start_factor=.1, final_lr_factor=.5)
        trace = []
        for _ in range(budget * 3):
            trace.append(scheduler.prepare_step()["neck"])
            scheduler.advance()
        traces.append(trace)
    assert traces[0] == traces[1][:15]
    assert traces[0][2] == pytest.approx(1e-5)
    assert traces[1][-1] == pytest.approx(5e-6)


def test_hog_schedule_has_no_extra_pretrain_or_tail_epochs():
    assert [hog_multiplier(epoch, 3, 4) for epoch in range(10)] == [0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert [hog_multiplier(1, macro, 4) for macro in range(4)] == [.25, .5, .75, 1]
    with pytest.raises(ValueError):
        hog_multiplier(10, 0, 4)


@pytest.mark.parametrize("native,aux", [(0, 1), (1, 0), (float("nan"), 1), (1, float("inf"))])
def test_calibration_invalid_gradient_fails_closed(native, aux):
    with pytest.raises(FloatingPointError):
        calibration_coefficient(native, aux)


def test_calibration_uses_gradient_ratio_not_fixed_mu():
    mu = calibration_coefficient(400, 4)
    assert mu == pytest.approx(.5)
    assert mu * (4 / 400) ** .5 == pytest.approx(.05)


def test_scaled_calibration_probe_preserves_tiny_fp16_gradients():
    raw = torch.tensor([1.0], dtype=torch.float16, requires_grad=True)
    loss = (raw.float() * 1e-9).sum()
    unscaled = torch.autograd.grad(loss, raw, retain_graph=True)[0]
    assert unscaled.item() == 0
    protected = probe_gradient(loss, raw)
    assert protected.dtype == torch.float32
    assert protected.item() == pytest.approx(1e-9, rel=.05)


def test_fitting_batch_without_vram_headroom_is_not_eligible():
    gib = 1024**3
    assert not memory_headroom(24*gib, 24*gib, 22*gib)["eligible"]
    assert memory_headroom(24*gib, 24*gib, 19*gib)["eligible"]
    assert not memory_headroom(24*gib, 20*gib, 18*gib)["eligible"]


def test_regression_pause_tracks_each_ap_not_only_joint_average():
    parent = {name: .5 for name in GATE_METRICS}
    candidate = dict(parent)
    candidate[GATE_METRICS[0]] = .495
    assert not severe_regressions(candidate, parent)
    candidate[GATE_METRICS[7]] = .4949
    assert set(severe_regressions(candidate, parent)) == {GATE_METRICS[7]}


class ToyMASF(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(2.0))

    def forward(self, value):
        return value * (1 + self.alpha)


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
        self.p3_masf = ToyMASF()

    def forward(self, value):
        return self.p3_masf(self.conv(value))


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph = nn.Module()
        self.graph.model = nn.ModuleList([nn.Identity() for _ in range(16)] + [ToyLayer()])

    def forward(self, image):
        return self.graph.model[16](image)

    def contract(self):
        return {"model_kind": "toy_base"}


class ToyAux(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(2, 9, 1)
        self.bins, self.cell_size = 9, 8
        self.calls = 0

    def forward(self, raw, images, boxes, indices):
        self.calls += 1
        loss = self.projection(raw).square().flatten(1).mean(1)
        return SimpleNamespace(loss=loss.mean(), loss_sum=loss.sum(), per_image_loss=loss,
            targets=SimpleNamespace(valid_mask=torch.ones(raw.shape[0], *raw.shape[-2:], dtype=torch.bool)))


class ToyNative:
    def __init__(self, base):
        self.base, self.updates = base, 0

    def loss_for(self, task, batch):
        value = self.base(batch["img"]).square().flatten(1).mean(1).sum()
        return TaskLossResult(Task(task), value, value.detach().reshape(1), batch["img"].shape[0])

    def advance_epoch(self, tasks=None):
        self.updates += 1

    def state_dict(self):
        return {"updates": self.updates}

    def load_state_dict(self, state):
        self.updates = state["updates"]


def test_validation_boundary_restores_rng_and_rejects_source_mutation():
    model = ToyBase()
    torch_before, python_before, numpy_before = torch.get_rng_state(), random.getstate(), np.random.get_state()
    with validation_boundary(model):
        torch.rand(7)
        random.random()
        np.random.rand(7)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:]
    with pytest.raises(RuntimeError, match="來源 state"):
        with validation_boundary(model):
            torch.rand(3)
            with torch.no_grad():
                next(model.parameters()).add_(1)
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize("adopt_prefix,pause_live", [(False, False), (True, False), (False, True)])
def test_live_validation_never_selects_checkpoints_or_pauses_healthy_ema(tmp_path, monkeypatch, adopt_prefix, pause_live):
    import yolo_optimize.training as training
    baseline = {name: .5 for name in GATE_METRICS}
    ema_metrics = {name: .51 for name in GATE_METRICS}
    live_metrics = {name: .1 for name in GATE_METRICS}
    model = TrainingModel(ToyBase())
    metadata = {"parent": "parent.pt", "criteria_continuation": {"parent_metrics": baseline}}
    state = SimpleNamespace(model=model, ema=SimpleNamespace(ema=copy.deepcopy(model)),
        metadata=metadata, router=SimpleNamespace(calibration=None),
        detect=SimpleNamespace(loader=[]), pose=SimpleNamespace(loader=[]), scheduler=None,
        guard=SimpleNamespace(assert_unchanged=lambda model: None), training_mode=lambda: None)
    config = SimpleNamespace(run_root=tmp_path, maximum_map_drop=.08, gradient_statistics_interval=100,
        preflight=lambda: SimpleNamespace(ready=True, baseline=baseline))
    selected_for_save, build_modes, trained_epochs, boundaries = [], [], [], []

    def fake_build(*args, **kwargs):
        build_modes.append(kwargs["ema_age_mode"])
        return state

    @dataclass
    class Report:
        next_global_macro_step: int

    class Runner:
        def __init__(self, **kwargs):
            pass

        def run_epoch(self, *, epoch, global_macro_step, stage):
            trained_epochs.append((epoch, global_macro_step))
            return Report(global_macro_step + 1)

    def fake_save(_state, _root, _epoch, _global, _metadata, _selectors, selection, metrics, _early, _seeds):
        selected_for_save.append((dict(metrics), dict(selection.scores)))

    monkeypatch.setattr(training, "_new_run", lambda path: tmp_path)
    monkeypatch.setattr(training, "_build", fake_build)
    monkeypatch.setattr(training, "_reseed", lambda state, epoch: {})
    monkeypatch.setattr(training, "JointEpochRunner", Runner)
    monkeypatch.setattr(training, "ExperimentLogger", lambda *args, **kwargs: nullcontext(None))
    monkeypatch.setattr(training, "_save_epoch", fake_save)
    def pending(_state, _root, epoch, *args):
        boundaries.append((epoch, "saved_pending"))
        return {}

    def validate(_state, _config, _data, _root, epoch, _sink, *, live=False):
        assert (epoch, "saved_pending") in boundaries
        boundaries.append((epoch, "validation"))
        return {"bittrue": live_metrics} if live else {"float": ema_metrics, "bittrue": ema_metrics}

    monkeypatch.setattr(training, "_save_pending_validation", pending)
    monkeypatch.setattr(training.runtime, "prepare_data", lambda *args: (Path("detect.yaml"), Path("pose.yaml")))
    monkeypatch.setattr(training.runtime, "write_json", lambda *args: None)
    monkeypatch.setattr(training, "_validate", validate)
    if adopt_prefix:
        from yolo_optimize import control_prefix
        monkeypatch.setattr(control_prefix, "adopt_ema_prefix", lambda *args: {
            "start_epoch": 1, "global_macro_step": 463,
            "metrics": {"float": ema_metrics, "bittrue": ema_metrics},
            "live_metrics": live_metrics, "training": {"next_global_macro_step": 463},
            "loader_state": {"seeds": {"detect": 0, "pose": 1}},
            "provenance": {"kind": "explicit_ema_prefix_adoption"}})
    result = training.run_training(config=config, checkpoint="parent.pt", paired_snapshot="paired.pt",
        run_dir=tmp_path, device="cpu", ema_age_mode="parent", validate_live=True,
        adopt_ema_prefix_dir=tmp_path / "diagnostic" if adopt_prefix else None,
        pause_on_live_regression=pause_live)
    assert build_modes == ["parent"]
    assert result["status"] == ("paused_for_analysis" if pause_live else "complete")
    assert len(result["epochs"]) == (1 if pause_live else 5)
    assert all(item["live_metrics"] == live_metrics for item in result["epochs"])
    assert all(set(item["live_regressions"]) == set(GATE_METRICS) for item in result["epochs"])
    assert all(values == ema_metrics and scores["best_joint"] == pytest.approx(.51)
               for values, scores in selected_for_save)
    assert result["best"]["best_joint"]["metrics"] == ema_metrics
    assert [epoch for epoch, _ in trained_epochs] == ([1] if pause_live else
        [2, 3, 4, 5] if adopt_prefix else [1, 2, 3, 4, 5])
    assert trained_epochs[0][1] == (463 if adopt_prefix else 0)
    assert len(boundaries) == len(trained_epochs) * 3
    if adopt_prefix:
        assert result["epochs"][0]["reused_prefix"] is True
        assert result["metadata"]["epochs_trained_in_this_process"] == 4
    if pause_live:
        assert len(selected_for_save) == 1
        assert result["regressions"] == {}
        assert set(result["live_regressions"]) == set(GATE_METRICS)


def batch(images):
    return {"img": images, "bboxes": torch.empty(0, 4), "batch_idx": torch.empty(0)}


def test_hook_captures_pre_masf_and_removes_on_failure():
    model = ToyBase()
    value = torch.randn(2, 3, 4, 4)
    with capture_raw_p3(model) as captured:
        output = model(value)
    torch.testing.assert_close(output, captured[0] * 3)
    assert captured[0].requires_grad
    assert not model.graph.model[16].p3_masf._forward_pre_hooks
    with pytest.raises(RuntimeError):
        with capture_raw_p3(model):
            raise RuntimeError("試驗錯誤")
    assert not model.graph.model[16].p3_masf._forward_pre_hooks


def test_router_batch_sum_preserves_microbatch_gradients():
    torch.manual_seed(7)
    model = TrainingModel(ToyBase(), ToyAux())
    images = torch.randn(4, 3, 4, 4)
    results = []
    for chunks in (1, 2):
        candidate = copy.deepcopy(model)
        router = HOGTaskLossRouter(ToyNative(candidate.base), candidate, device="cpu", amp=False)
        router.mu = .3
        for part in images.chunk(chunks):
            result = router.loss_for("detect", batch(part))
            (result.raw_total / len(images)).backward()
        results.append([parameter.grad.clone() for parameter in candidate.parameters()])
    for whole, split in zip(*results, strict=True):
        torch.testing.assert_close(whole, split)


def test_mu_zero_does_not_apply_aux_weight_decay():
    model = TrainingModel(ToyBase(), ToyAux())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.3)
    router = HOGTaskLossRouter(ToyNative(model.base), model, device="cpu", amp=False)
    before = {name: value.clone() for name, value in model.aux.state_dict().items()}
    router.loss_for("detect", batch(torch.randn(2, 3, 4, 4))).raw_total.backward()
    assert all(parameter.grad is None for parameter in model.aux.parameters())
    optimizer.step()
    assert model.aux.calls == 0
    for name, value in model.aux.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_scope_freezes_masf_backbone_attention_and_shared_bn():
    model = ToyBase()
    model.graph.model[0] = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
    model.graph.model[16].bn = nn.BatchNorm2d(2)
    model.graph.model[16].attn = nn.Linear(2, 2)
    optimizer, _ = build_joint_optimizer(model, SCOPE)
    apply_stage(model, SCOPE)
    names = dict(model.named_parameters())
    assert names["graph.model.16.conv.weight"].requires_grad
    assert not names["graph.model.16.p3_masf.alpha"].requires_grad
    assert not names["graph.model.0.0.weight"].requires_grad
    assert not names["graph.model.16.attn.weight"].requires_grad
    assert not model.graph.model[16].bn.training
    assert model.graph.model[16].bn.weight.requires_grad
    assert optimizer.defaults["betas"] == (.948, .999)


def test_full_snapshot_contains_aux_but_inference_is_stripped(tmp_path):
    model = TrainingModel(ToyBase(), ToyAux())
    parameters = list(model.named_parameters())
    optimizer = torch.optim.AdamW([{"params": [p for _, p in parameters],
        "param_names": tuple(name for name, _ in parameters), "group_name": "all", "role": "toy", "lr": .01}])
    ema = ModelEMA(model)
    scheduler = StageWarmupCosineScheduler(optimizer, stage="recovery", epochs=10,
                                         steps_per_epoch=1, warmup_epochs=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    router = HOGTaskLossRouter(ToyNative(model.base), model, device="cpu", amp=False)
    snapshot = tmp_path / "full.pt"
    save_training_snapshot(snapshot, model=model, ema=ema, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, criteria=router,
        progress=TrainingProgress("recovery", 1, 1, 1), resolved_config={}, provenance={},
        loader_state={}, best_state={})
    full = torch.load(snapshot, weights_only=True)
    assert any(name.startswith("aux.") for name in full["model_state"])
    assert full["contract"]["auxiliary"]["kind"] == "raw_p3_hog"
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3)
    load_training_snapshot(snapshot, model=model, ema=ema, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, criteria=router, restore_rng=False)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    inference = tmp_path / "inference.pt"
    save_inference_weights(inference, model=model.base, ema=SimpleNamespace(ema=ema.ema.base))
    payload = torch.load(inference, weights_only=True)
    assert payload["contract"] == model.base.contract()
    assert set(payload["state_dict"]) == set(model.base.state_dict())
    assert not any(name.startswith("aux.") for name in payload["state_dict"])


def test_pending_validation_snapshot_is_resumable_but_not_a_new_best(tmp_path):
    from yolo_optimize.training import _save_pending_validation
    from yolo_combine.metrics import CheckpointSelectors
    model = TrainingModel(ToyBase())
    named = list(model.named_parameters())
    optimizer = torch.optim.AdamW([{"params": [p for _, p in named],
        "param_names": tuple(n for n, _ in named), "group_name": "all", "role": "toy", "lr": .01}])
    ema = ModelEMA(model)
    scheduler = StageWarmupCosineScheduler(optimizer, stage="recovery", epochs=10,
        steps_per_epoch=1, warmup_epochs=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    router = HOGTaskLossRouter(ToyNative(model.base), model, device="cpu", amp=False)
    state = SimpleNamespace(model=model, ema=ema, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, router=router)
    selectors = CheckpointSelectors()
    saved = _save_pending_validation(state, tmp_path, 0, 1, {"parent": "parent.pt"}, selectors,
        None, {"detect": 0, "pose": 1})
    payload = torch.load(saved["path"], weights_only=True)
    assert payload["checkpoint_kind"] == "full_resume"
    assert payload["loader_state"]["validation_pending"] is True
    assert payload["progress"]["next_epoch"] == 1
    assert payload["best_state"] == selectors.state_dict()
    assert not (tmp_path / "inference").exists()
    restored = load_training_snapshot(saved["path"], model=model, ema=ema, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, criteria=router, restore_rng=True)
    assert restored.loader_state["snapshot_boundary"] == "epoch_end_before_validation"
