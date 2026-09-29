from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from achitechure_2.profiling import (
    _gflops,
    _load_inference_checkpoint,
    _percentile,
    _prepare_profile_model,
)


class _ProfileToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, images: torch.Tensor, task: str) -> dict[str, torch.Tensor]:
        return {task: self.conv(images)}

    def contract(self) -> dict[str, object]:
        return {"kind": "profile-toy", "channels": 4}

class _UnregisteredCacheToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.inference_cache = torch.empty(0)

    def forward(self, images: torch.Tensor, task: str) -> dict[str, torch.Tensor]:
        if self.inference_cache.numel() and self.inference_cache.device != images.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least "
                f"two devices, {images.device} and {self.inference_cache.device}!"
            )
        self.inference_cache = torch.ones(1, device=images.device)
        return {task: images * self.weight}



def _checkpoint(path: Path, model: _ProfileToy, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "checkpoint_kind": "inference_only",
        "contract": model.contract(),
        "source": "ema",
        "state_dict": model.state_dict(),
        "metadata": {"candidate": "C0"},
    }
    payload.update(overrides)
    torch.save(payload, path)
    return path


def test_percentile_interpolates_and_rejects_empty_values() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError, match="不得為空"):
        _percentile([], 0.95)


def test_inference_checkpoint_load_is_strict_and_requires_ema(tmp_path: Path) -> None:
    source = _ProfileToy()
    target = _ProfileToy()
    path = _checkpoint(tmp_path / "ema.pt", source)

    payload = _load_inference_checkpoint(target, path)

    assert payload["metadata"] == {"candidate": "C0"}
    assert all(
        torch.equal(left, right)
        for left, right in zip(source.state_dict().values(), target.state_dict().values())
    )

    live = _checkpoint(tmp_path / "live.pt", source, source="live")
    with pytest.raises(ValueError, match="不支援"):
        _load_inference_checkpoint(_ProfileToy(), live)

    drifted = _checkpoint(
        tmp_path / "drifted.pt",
        source,
        contract={"kind": "wrong"},
    )
    with pytest.raises(ValueError, match="contract"):
        _load_inference_checkpoint(_ProfileToy(), drifted)


def test_gflops_is_positive_and_route_specific_wrapper_is_callable() -> None:
    model = _ProfileToy().eval()

    result = _gflops(model, task="detect", imgsz=32)

    assert result > 0

def test_prepare_profile_model_builds_dynamic_cache_on_final_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited: list[torch.device] = []

    def fake_gflops(model: nn.Module, *, task: str, imgsz: int) -> float:
        device = next(model.parameters()).device
        visited.append(device)
        model(
            torch.zeros(1, 3, imgsz, imgsz, device=device),
            task=task,
        )
        return 1.0

    monkeypatch.setattr("achitechure_2.profiling._gflops", fake_gflops)
    model, task_flops = _prepare_profile_model(
        _UnregisteredCacheToy().eval(),
        device=torch.device("meta"),
        imgsz=4,
    )

    assert visited == [torch.device("meta")] * 3
    assert model.inference_cache.device == torch.device("meta")
    assert task_flops == {"detect": 1.0, "pose": 1.0, "both": 1.0}
    model(torch.zeros(1, 3, 4, 4, device="meta"), task="detect")
