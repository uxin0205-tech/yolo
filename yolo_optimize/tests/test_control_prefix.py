"""小型 CPU full snapshot 的真實 save/load 回歸；不使用正式模型或 GPU。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_optimize import runtime
from yolo_optimize.control_prefix import adopt_ema_prefix, _digest_state
from yolo_optimize.training_safety import FixedStateEMA
from yolo_combine.joint_trainer import StageWarmupCosineScheduler
from yolo_combine.metrics import AccuracyGate, CheckpointSelectors, GATE_METRICS, joint_score
from yolo_combine.resume import TrainingProgress, save_training_snapshot


class PrefixModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Linear(2, 1))
        self.base.fixed = nn.Parameter(torch.tensor([1.234567]), requires_grad=False)
        self.aux = None

    def contract(self):
        return {"model_kind": "direction1_training", "auxiliary": None, "fixture": "prefix_cpu"}


class PrefixCriteria:
    def __init__(self):
        self.native = {}
        for task, updates in (("detect", 1), ("pose", 2)):
            o2m = max(1 - updates / 9, 0) * (0.8 - 0.1) + 0.1
            self.native[task] = {"end2end": True, "updates": updates, "o2m": o2m,
                                 "o2o": 1 - o2m, "o2m_copy": 0.8, "final_o2m": 0.1, "total": 1.0}

    def state_dict(self):
        return {"schema_version": 1, "native": deepcopy(self.native), "mu": 0.0, "calibration": None}

    def load_state_dict(self, state):
        self.native = deepcopy(state["native"])

    def advance_epoch(self):
        for state in self.native.values():
            state["updates"] += 1
            state["o2m"] = max(1 - state["updates"] / 9, 0) * (0.8 - 0.1) + 0.1
            state["o2o"] = 1 - state["o2m"]


class ControlPrefixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent = self.root / "parent-ema.pt"
        self.parent.write_bytes(b"immutable parent identity fixture")
        torch.manual_seed(300)
        self.initial = PrefixModel().train()
        self.parent_metrics = {key: 0.5 for key in GATE_METRICS}
        self.parent_age = 26597
        self.steps = 3  # CPU 小型 epoch；正式 state.macros=463，同樣校驗相符才可採用。
        trained = self.new_state()
        for _ in range(self.steps):
            trained.scheduler.prepare_step()
            loss = trained.model.base(torch.randn(4, 2)).square().mean()
            loss.backward()
            trained.optimizer.step()
            trained.optimizer.zero_grad(set_to_none=True)
            trained.ema.update(trained.model)
            trained.scheduler.advance()
        trained.router.advance_epoch()
        self.trained = trained
        source = deepcopy(trained.metadata)
        source["ema"]["updates_at_start"] = 0
        source["ema"].pop("age_mode")
        source["criteria_continuation"].pop("ema_updates")
        source.update(experiment="paired_ema_age", max_epochs=1, calibration=None,
                      parent_sha256=runtime.sha256(self.parent), parent_ema_updates=self.parent_age)
        self.snapshot = self.root / "checkpoints" / "continued-epoch-0001.pt"
        saved = save_training_snapshot(
            self.snapshot, model=trained.model, ema=trained.ema, optimizer=trained.optimizer,
            scheduler=trained.scheduler, scaler=trained.scaler, criteria=trained.router,
            progress=TrainingProgress(stage="ema_age_diagnostic", next_epoch=1,
                                      global_macro_step=self.steps, joint_epochs_completed=1),
            resolved_config={**source, "ema_variant": "continued"}, provenance={"diagnostic_only": True},
            loader_state={"snapshot_boundary": "epoch_end_before_validation",
                          "seeds": {"detect": 6148914691236517205, "pose": 6148914691236517206}},
            best_state={"automatic_promotion": False},
        )
        metrics = {key: 0.501 for key in GATE_METRICS}
        self.summary = {
            "status": "complete", "shared_live_trajectory": True, "independent_replicates": False,
            "automatic_promotion": False, "hog_enabled": False, "metadata": source,
            "snapshots": {"continued": {"path": str(self.snapshot), "sha256": saved.sha256, "bytes": saved.bytes}},
            "ema_observers_at_end": {"current_updates": {"fresh": self.steps, "continued": self.parent_age + self.steps},
                "observations": self.steps, "fixed_state_names": list(trained.ema.fixed_state_names)},
            "epochs": [{"epoch": 1, "training": {"epoch": 1, "stage": "ema_age_diagnostic", "macros": self.steps,
                                                   "next_global_macro_step": self.steps}}],
            "versions": {
                "continued": {"source": "ema", "state_sha256": _digest_state(trained.ema.ema.base.state_dict()),
                              "metrics": {"float": metrics, "bittrue": metrics},
                              "joint_scores": {"float": joint_score(metrics), "bittrue": joint_score(metrics)}},
                "live": {"source": "live", "state_sha256": _digest_state(trained.model.base.state_dict()),
                         "metrics": {"bittrue": metrics}},
            },
        }
        self.write_summary()

    def new_state(self):
        model = deepcopy(self.initial)
        parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW([{"params": [value for _, value in parameters],
            "param_names": tuple(name for name, _ in parameters), "group_name": "neck.decay", "role": "neck"}],
            lr=1e-5, betas=(0.948, 0.999), eps=1e-8, weight_decay=0.00027)
        scheduler = StageWarmupCosineScheduler(optimizer, stage="recovery", epochs=10,
            steps_per_epoch=self.steps, warmup_epochs=1, warmup_start_factor=0.1, final_lr_factor=0.5)
        ema = FixedStateEMA(model, updates=self.parent_age)
        router = PrefixCriteria()
        paths = [str(self.root / "coco.yaml"), str(self.root / "pose.yaml")]
        metadata = {"variant": "native", "parent": str(self.parent),
            "factory": {"names": {0: "fixture"}, "shape": (2, 1)},
            "loaded": {"path": str(self.parent), "state_source": "ema"},
            "scope": {"name": "fixture_native_neck_heads", "warmup_epochs": 1},
            "optimizer": {"name": "AdamW"}, "optimizer_hyperparameters": {"name": "AdamW", "fresh_state": True},
            "loaders": {"detect": {"batch": 32}, "pose": {"batch": 16}}, "amp": {"requested": False},
            "gradient_clip_norm": 10.0, "scheduler": {"steps_per_epoch": self.steps, "horizon_epochs": 10},
            "physical_batch": 32, "detect_logical": 128, "detect_per_macro": 256, "pose_per_macro": 16,
            "reference_batch": 64, "seed": 0, "scheduler_horizon": 10, "warmup_epochs": 1,
            "initial_native_criterion": deepcopy(router.native), "base_state_sha256": _digest_state(model.base.state_dict()),
            "data": paths, "deployment_auxiliary": False,
            "ema": {"age_mode": "parent", "updates_at_start": self.parent_age, "decay": 0.9999, "tau": 2000,
                    "fixed_state_exact_copy": True, "fixed_state_names": list(ema.fixed_state_names)},
            "criteria_continuation": {"path": str(self.parent), "horizons": {"detect": 10, "pose": 10},
                "state": deepcopy(router.native), "parent_metrics": self.parent_metrics, "parent_progress": {"epoch": 58},
                "ema_updates": self.parent_age},
        }
        return SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema, router=router,
            scaler=torch.amp.GradScaler("cpu", enabled=False), metadata=metadata, macros=self.steps,
            detect=SimpleNamespace(data_yaml=Path(paths[0])), pose=SimpleNamespace(data_yaml=Path(paths[1])))

    def write_summary(self):
        (self.root / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def rewrite_snapshot(self, mutate):
        payload = torch.load(self.snapshot, map_location="cpu", weights_only=True)
        mutate(payload)
        torch.save(payload, self.snapshot)
        self.summary["snapshots"]["continued"].update(
            sha256=runtime.sha256(self.snapshot), bytes=self.snapshot.stat().st_size)
        self.write_summary()

    def test_real_full_resume_restores_training_state_and_returns_selector_input(self):
        state = self.new_state()
        # 真正的 _build metadata 尚未合併 data；應核對目前兩個 loader 的路徑。
        state.metadata.pop("data")
        original_sha = runtime.sha256(self.snapshot)
        payload = torch.load(self.snapshot, map_location="cpu", weights_only=True)
        torch.manual_seed(999)
        result = adopt_ema_prefix(state, self.root, self.parent_metrics)
        self.assertTrue(torch.equal(torch.get_rng_state(), payload["rng"]["torch_cpu"]))
        self.assertEqual(result["start_epoch"], 1)
        self.assertEqual(result["global_macro_step"], self.steps)
        self.assertEqual(result["loader_state"]["seeds"], payload["loader_state"]["seeds"])
        self.assertTrue(result["reused"])
        self.assertFalse(result["provenance"]["failed_run_exact_state_equivalence_claimed"])
        self.assertEqual(state.ema.updates, self.parent_age + self.steps)
        self.assertEqual(state.scheduler.current_step, self.steps)
        self.assertEqual(state.router.state_dict(), self.trained.router.state_dict())
        for label, model in (("model_state", state.model), ("ema_state", state.ema.ema)):
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, payload[label][name], rtol=0, atol=0)
        restored_optimizer = state.optimizer.state_dict()
        self.assertEqual(restored_optimizer["param_groups"], payload["optimizer_state"]["param_groups"])
        for identifier, values in restored_optimizer["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, payload["optimizer_state"]["state"][identifier][name], rtol=0, atol=0)
        parameters = [parameter for group in state.optimizer.param_groups for parameter in group["params"]]
        self.assertTrue(all(parameter in state.optimizer.state for parameter in parameters))
        self.assertTrue(all(isinstance(key, nn.Parameter) for key in state.optimizer.state))
        # 下一個真實 AdamW step 必須與不中斷 reference 相同，不能只有序列化字典看似一致。
        next_input = torch.tensor([[0.1, 0.2], [0.3, -0.4], [-0.2, 0.1], [0.8, 0.7]])
        for candidate in (state, self.trained):
            candidate.scheduler.prepare_step()
            candidate.model.base(next_input).square().mean().backward()
            candidate.optimizer.step()
            candidate.optimizer.zero_grad(set_to_none=True)
            candidate.ema.update(candidate.model)
            candidate.scheduler.advance()
        for left, right in ((state.model, self.trained.model), (state.ema.ema, self.trained.ema.ema)):
            for name, value in left.state_dict().items():
                torch.testing.assert_close(value, right.state_dict()[name], rtol=0, atol=0)
        for left, right in zip(parameters, self.trained.optimizer.param_groups[0]["params"], strict=True):
            for name in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(state.optimizer.state[left][name],
                    self.trained.optimizer.state[right][name], rtol=0, atol=0)
        selector = CheckpointSelectors()
        observed = selector.observe(epoch=1, metrics=result["metrics"]["bittrue"],
            gate=AccuracyGate(self.parent_metrics).evaluate(result["metrics"]["bittrue"]))
        self.assertIn("best_joint", observed.selected)
        self.assertEqual(runtime.sha256(self.snapshot), original_sha)

    def test_parent_identity_mismatch_is_rejected_before_restore(self):
        self.summary["metadata"]["parent_sha256"] = "0" * 64
        self.write_summary()
        state = self.new_state()
        with self.assertRaisesRegex(ValueError, "parent SHA"):
            adopt_ema_prefix(state, self.root, self.parent_metrics)
        self.assertFalse(state.optimizer.state)
        self.assertEqual(state.scheduler.current_step, 0)

    def test_mismatched_scheduler_horizon_or_step_is_rejected_before_restore(self):
        pristine = self.snapshot.read_bytes()
        for key, wrong_value in (("epochs", 5), ("current_step", self.steps - 1)):
            with self.subTest(key=key):
                self.snapshot.write_bytes(pristine)
                self.rewrite_snapshot(lambda payload: payload["scheduler_state"].update({key: wrong_value}))
                state = self.new_state()
                with self.assertRaisesRegex(ValueError, "snapshot scheduler horizon/step"):
                    adopt_ema_prefix(state, self.root, self.parent_metrics)
                self.assertFalse(state.optimizer.state)

    def test_missing_e1_selector_metric_is_not_silently_accepted(self):
        del self.summary["versions"]["continued"]["metrics"]["bittrue"][GATE_METRICS[0]]
        self.write_summary()
        with self.assertRaisesRegex(ValueError, "selector 必要八 AP"):
            adopt_ema_prefix(self.new_state(), self.root, self.parent_metrics)

    def test_native_loader_rejects_optimizer_manifest_mismatch(self):
        self.rewrite_snapshot(lambda payload: payload["optimizer_manifest"][0].update(group_name="wrong.group"))
        state = self.new_state()
        with self.assertRaisesRegex(ValueError, "optimizer parameter manifest changed"):
            adopt_ema_prefix(state, self.root, self.parent_metrics)
        self.assertFalse(state.optimizer.state)


if __name__ == "__main__":
    unittest.main()
