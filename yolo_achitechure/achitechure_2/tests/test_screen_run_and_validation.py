from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset

from achitechure_2.runtime_dataset import RuntimeLabelCacheYOLODataset
from achitechure_2.screen_training import ScreenRunConfig, probe_screen_memory
from achitechure_2.screen_validation import (
    ThresholdSet,
    _metric_payload,
    _RuntimeCacheValidatorMixin,
)


def _isolated_run_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pose: Any) -> Path:
    payload = yaml.safe_load(Path("configs/runs/full35-float-screen-20.yaml").read_text(encoding="utf-8"))
    payload["authorization"]["pose"] = pose
    payload["training"]["pose_enabled"] = pose
    files = {
        "manifest": tmp_path / "handoff.json",
        "screen_manifest": tmp_path / "screening.json",
        "detect": tmp_path / "detect.yaml",
        "pose": tmp_path / "pose.yaml",
        "diagnostic_detect": tmp_path / "diagnostic-detect.yaml",
    }
    for path in files.values():
        path.write_text("placeholder\n", encoding="utf-8")
    screen_root = tmp_path / "screen-root"
    screen_root.mkdir()
    payload["handoff"]["manifest"] = str(files["manifest"])
    payload["datasets"].update(
        {
            "screen_manifest": str(files["screen_manifest"]),
            "screen_root": str(screen_root),
            "detect": str(files["detect"]),
            "pose": str(files["pose"]),
            "diagnostic_detect": str(files["diagnostic_detect"]),
        }
    )
    monkeypatch.setattr(
        "achitechure_2.screen_training.validate_screening_data",
        lambda _: {"coco_train_count": 23657, "bbat5_train_count": 1073},
    )
    destination = tmp_path / "run.yaml"
    destination.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination


def test_screen_run_pose_gate_requires_explicit_boolean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = _isolated_run_yaml(tmp_path, monkeypatch, "pending_user_decision")
    with pytest.raises(PermissionError, match="Pose gate"):
        ScreenRunConfig.load(pending)


@pytest.mark.parametrize("pose_enabled", [False, True])
def test_screen_run_supports_user_selected_detect_only_or_joint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pose_enabled: bool,
) -> None:
    path = _isolated_run_yaml(tmp_path, monkeypatch, pose_enabled)
    config = ScreenRunConfig.load(path)

    assert config.pose_enabled is pose_enabled
    assert config.detect_logical_batch == 128
    assert config.detect_microbatch == 32
    assert config.detect_oom_fallback_microbatch == 16
    assert config.detect_val_batch == config.pose_val_batch == 16


def test_memory_probe_initializes_runtime_before_resetting_cuda_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    device = object()
    runtime = SimpleNamespace(
        device=device,
        detect_loader=SimpleNamespace(loader=[{"img": "detect"}]),
        pose_loader=None,
        microbatches_per_macro=1,
        model=object(),
        guard=SimpleNamespace(assert_unchanged=lambda _: events.append("guard")),
    )
    config = SimpleNamespace(
        detect_logical_batch=128,
        pose_enabled=False,
        pose_batch=16,
    )
    engine = SimpleNamespace(
        run=lambda **_: SimpleNamespace(detect_images=1, pose_images=0)
    )

    def make_runtime(*_: Any, **__: Any) -> Any:
        events.append("runtime")
        return runtime

    monkeypatch.setattr("achitechure_2.screen_training._make_runtime", make_runtime)
    monkeypatch.setattr("achitechure_2.screen_training._engine", lambda *_: engine)
    monkeypatch.setattr(
        "achitechure_2.screen_training._apply_training_mode",
        lambda _: events.append("training_mode"),
    )
    monkeypatch.setattr(
        "achitechure_2.screen_training.torch.cuda.empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(
        "achitechure_2.screen_training.torch.cuda.reset_peak_memory_stats",
        lambda value: events.append(f"reset:{value is device}"),
    )
    monkeypatch.setattr(
        "achitechure_2.screen_training.torch.cuda.synchronize", lambda _: None
    )
    monkeypatch.setattr(
        "achitechure_2.screen_training.torch.cuda.max_memory_allocated",
        lambda value: 2**20 if value is device else 0,
    )
    monkeypatch.setattr(
        "achitechure_2.screen_training.torch.cuda.max_memory_reserved",
        lambda value: 2 * 2**20 if value is device else 0,
    )

    report = probe_screen_memory(config, microbatch=32)

    assert events.index("runtime") < events.index("reset:True")
    assert report["peak_allocated_mib"] == 1
    assert report["peak_reserved_mib"] == 2


def test_runtime_label_cache_ignores_source_adjacent_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = object.__new__(RuntimeLabelCacheYOLODataset)
    dataset.runtime_label_cache_path = tmp_path / "runtime" / "labels.cache"
    dataset.runtime_label_cache_hit = False
    dataset.label_files = ["/read-only/labels/a.txt"]
    dataset.im_files = ["/read-only/images/a.jpg"]
    used: list[Path] = []

    def missing(_: Path) -> dict[str, Any]:
        raise FileNotFoundError

    def cache_labels(_: YOLODataset, path: Path) -> dict[str, Any]:
        used.append(path)
        return {"version": DATASET_CACHE_VERSION}

    monkeypatch.setattr("achitechure_2.runtime_dataset.load_dataset_cache_file", missing)
    monkeypatch.setattr(YOLODataset, "cache_labels", cache_labels)
    result = dataset.cache_labels(Path("/read-only/labels.cache"))

    assert result["version"] == DATASET_CACHE_VERSION
    assert used == [dataset.runtime_label_cache_path]
    assert dataset.runtime_label_cache_path.parent.is_dir()


def test_validator_routes_label_index_to_runtime_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = object.__new__(_RuntimeCacheValidatorMixin)
    validator._label_cache_path = tmp_path / "runtime" / "detect-val.cache"
    validator.args = SimpleNamespace(batch=16)
    validator.data = {"names": {0: "person"}}
    validator.stride = 32
    captured: dict[str, Any] = {}

    def fake_build(*args: Any, **kwargs: Any) -> str:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "dataset"

    monkeypatch.setattr(
        "achitechure_2.screen_validation.build_runtime_yolo_dataset",
        fake_build,
    )
    result = validator.build_dataset("/read-only/images.txt", batch=None)

    assert result == "dataset"
    assert captured["args"][1] == "/read-only/images.txt"
    assert captured["args"][2] == 16
    assert captured["kwargs"]["label_cache_path"] == validator._label_cache_path
    assert captured["kwargs"]["mode"] == "val"
    assert captured["kwargs"]["stride"] == 32


class _Metric:
    ap_class_index = np.array([0, 1])
    px = np.array([0.0, 0.5, 1.0])
    p_curve = np.array([[0.2, 0.8, 1.0], [0.3, 0.6, 1.0]])
    r_curve = np.array([[1.0, 0.5, 0.0], [1.0, 0.75, 0.0]])
    f1_curve = 2 * p_curve * r_curve / np.maximum(p_curve + r_curve, 1e-16)
    all_ap = np.array([[0.7] + [0.4] * 9, [0.6] + [0.3] * 9])
    map50 = 0.65
    map = 0.38
    mp = 0.7
    mr = 0.625


def test_validation_keeps_ap50_ap50_95_macro_and_micro_f1_separate() -> None:
    payload, threshold = _metric_payload(
        _Metric(),
        names={0: "ball", 1: "bat"},
        supports=np.array([10, 20]),
        threshold=None,
    )

    assert threshold == pytest.approx(0.5)
    assert payload["ap"]["map50"] == pytest.approx(0.65)
    assert payload["ap"]["map50_95"] == pytest.approx(0.38)
    assert payload["ap"]["per_class"]["0"]["ap50"] == pytest.approx(0.7)
    assert payload["ap"]["per_class"]["0"]["ap50_95"] == pytest.approx(0.43)
    assert payload["f1"]["macro_f1"] == pytest.approx(0.64102564)
    assert payload["f1"]["micro_f1"] == pytest.approx(0.65306122)
    assert (
        payload["f1"]["micro_f1_method"] == "estimated_from_precision_recall_curves_and_supports"
    )
    assert payload["f1"]["threshold_derived_in_this_event"] is True

    fixed, resolved = _metric_payload(
        _Metric(),
        names={0: "ball", 1: "bat"},
        supports=np.array([10, 20]),
        threshold=0.49,
    )
    assert resolved == pytest.approx(0.5)
    assert fixed["f1"]["threshold_derived_in_this_event"] is False
    assert (
        ThresholdSet.from_mapping({"detect_box": 0.5, "pose_box": 0.4, "pose_keypoints": 0.3}).detect_box
        == 0.5
    )
