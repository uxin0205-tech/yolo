from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/full35_profile.py"
SPEC = importlib.util.spec_from_file_location("full35_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
SPEC.loader.exec_module(PROFILE)


class _SharedActivationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.SiLU()
        self.first = nn.Sequential(nn.Identity(), shared)
        self.second = nn.Sequential(nn.Identity(), shared)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(value))


def test_profile_probes_keep_shared_activation_paths_separate() -> None:
    model = _SharedActivationModel()
    observer = PROFILE._Observer(bins=32, minimum=-16.0, maximum=16.0)
    PROFILE._install_probes(model, ("first.1", "second.1"), observer)

    model(torch.tensor([[-2.0, 0.0, 3.0]]))

    assert set(observer.states) == {"first.1", "second.1"}
    assert observer.states["first.1"].count == 3
    assert observer.states["second.1"].count == 3
    first = observer.states["first.1"].to_dict()
    assert first["nonfinite_count"] == 0
    assert first["negative_fraction"] == 1 / 3


class _RecordingLossRouter:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def loss_for(self, task: str, batch: dict[str, torch.Tensor]) -> None:
        self.tasks.append(task)
        assert "img" in batch


def test_pose_profile_uses_formal_loss_path() -> None:
    model = _SharedActivationModel()
    router = _RecordingLossRouter()

    PROFILE._profile_step(
        model,
        task="pose",
        batch={"img": torch.zeros(1, 1)},
        loss_router=router,
    )

    assert router.tasks == ["pose"]
