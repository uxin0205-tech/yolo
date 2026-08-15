from __future__ import annotations

from pathlib import Path

from yolo_attention.runner import TrainingRequest, launch_training

ROOT = Path(__file__).resolve().parents[1]


def request(tmp_path: Path) -> TrainingRequest:
    return TrainingRequest.from_files(
        variant_path=ROOT / "configs" / "variants" / "h-screen.yaml",
        training_path=ROOT / "configs" / "training" / "screening.yaml",
        artifacts_root=tmp_path,
        run_id="h-screen-seed0",
    )


def test_training_request_is_dry_run_serializable(tmp_path: Path) -> None:
    payload = request(tmp_path).summary()

    assert payload["run_id"] == "h-screen-seed0"
    assert payload["variant"] == "H-SCR"
    assert payload["stage"] == "screening"
    assert payload["will_execute"] is False


def test_launch_training_wires_custom_trainer_without_real_gpu(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            calls["weights"] = weights

        def train(self, *, trainer, **kwargs):
            calls["trainer"] = trainer
            calls["kwargs"] = kwargs
            return {"ok": True}

    result = launch_training(request(tmp_path), model_factory=FakeYOLO)

    assert result == {"ok": True}
    assert calls["weights"] == "weights/yolo26m.pt"
    assert calls["trainer"].training_stage == "screening"
    assert calls["kwargs"]["epochs"] == 10
    assert (tmp_path / "h-screen-seed0" / "manifest.json").is_file()
