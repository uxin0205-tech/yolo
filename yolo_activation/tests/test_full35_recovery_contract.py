from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from activation_lab.training import (
    Full35ActivationExperiment,
    Full35ExperimentConfig,
    load_full35_manifest,
    uniform_full35_policy,
)
from activation_lab.training.full35 import _fp32_bbox_iou, _RecoveryConfigProxy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"


@dataclass(frozen=True)
class _Preflight:
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    baseline: dict[str, float] | None


class _BaseConfig:
    maximum_map_drop = 0.08

    def preflight(self) -> _Preflight:
        return _Preflight(True, (), (), {"standalone": 1.0})

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline_metrics": "/release/standalone.yaml",
            "maximum_map_drop": self.maximum_map_drop,
        }


def test_recovery_uses_delivered_baseline_and_records_fresh_control_semantics() -> None:
    config = Full35ExperimentConfig.load(RECIPE)
    manifest = load_full35_manifest(config)
    proxy = _RecoveryConfigProxy(
        _BaseConfig(),
        recipe=config,
        phase=config.phase("short_recovery"),
        manifest=manifest,
        policy=uniform_full35_policy("silu"),
    )

    accepted = yaml.safe_load(config.baseline_metrics.read_text(encoding="utf-8"))
    report = proxy.preflight()
    payload = proxy.as_dict()
    activation = payload["activation_experiment"]

    assert report.baseline == accepted["metrics"]
    assert proxy.maximum_map_drop == 0.015
    assert payload["baseline_metrics"] == str(config.baseline_metrics)
    assert payload["maximum_map_drop"] == 0.015
    assert activation["initialization"] == {
        "model": "accepted_inference_ema_state_dict",
        "optimizer": "fresh",
        "scheduler": "fresh",
        "ema": "fresh_from_loaded_model",
        "rng": "reseeded_per_candidate",
    }
    assert activation["full_resume_checkpoint_role"] == (
        "lineage_and_emergency_exact_resume_only"
    )
    assert activation["numerical_stability"] == {
        "model_amp": True,
        "ciou_loss_precision": "fp32",
        "ciou_formula_changed": False,
    }


def test_fp32_ciou_wrapper_keeps_degenerate_box_backward_finite() -> None:
    config = Full35ExperimentConfig.load(RECIPE)
    experiment = Full35ActivationExperiment(config)
    experiment._imports()
    import ultralytics.utils.loss as loss_module

    stable = _fp32_bbox_iou(loss_module.bbox_iou)
    predicted = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0]], dtype=torch.float16, requires_grad=True
    )
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float16)

    value = stable(predicted, target, xywh=False, CIoU=True)
    value.sum().backward()

    assert value.dtype == torch.float32
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
