from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yolo_combine import _formal_training_impl as impl


class _ReachedOptimizerSetup(RuntimeError):
    """Sentinel proving formal startup passed macro-step calculation."""


def test_formal_startup_computes_macro_steps_before_optimizer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        architecture="full35",
        beta1=0.9,
        beta2=0.999,
        detect_batches_per_macro=2,
        detect_microbatches_per_macro=4,
        detect_data=tmp_path / "detect.yaml",
        enable_j3=False,
        optimizer="adamw",
        pose_checkpoint=tmp_path / "pose.pt",
        pose_data=tmp_path / "pose.yaml",
        seed=0,
        source_bundle=tmp_path / "source",
        stages=("j1",),
        weight_decay=5e-4,
        xnor_token_tile=32,
        preflight=lambda: SimpleNamespace(ready=True, baseline={}, blockers=()),
    )
    built = SimpleNamespace(model=torch.nn.Identity())
    monkeypatch.setattr(impl, "SourceBundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        impl,
        "FusionModelFactory",
        lambda *args, **kwargs: SimpleNamespace(build=lambda **build_kwargs: built),
    )

    def stop_after_macro_count(*args, **kwargs):
        raise _ReachedOptimizerSetup

    monkeypatch.setattr(impl, "build_joint_optimizer", stop_after_macro_count)
    session = impl.FormalJointTrainingSession.__new__(
        impl.FormalJointTrainingSession
    )
    session.config = config
    session.device = torch.device("cpu")
    session.run_name = "startup-regression"
    session.run_dir = tmp_path / "run"
    session.detect_microbatches_per_macro = 4
    detect_loader = SimpleNamespace(loader=(object(), object(), object()))
    pose_loader = SimpleNamespace(loader=(object(),))
    session._loaders = lambda model: (detect_loader, pose_loader, object())

    with pytest.raises(_ReachedOptimizerSetup):
        session.run()
