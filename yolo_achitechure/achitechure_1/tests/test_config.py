from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from achitechure_1.config import CommonTrainingConfig, load_phase_spec

ROOT = Path(__file__).resolve().parents[1]


def test_training_configs_disable_accumulation_and_match_fixed_phases() -> None:
    common = CommonTrainingConfig.from_yaml(ROOT / "configs/training/common.yaml")

    assert common.batch == 16 and common.nbs == 16
    assert common.optimizer == "MuSGD"
    assert common.imgsz == 640 and common.amp is True and common.workers == 8
    expected_patience = {"a1": 100, "a2": 4, "b": 4, "c": 8}
    for name, patience in expected_patience.items():
        phase = load_phase_spec(ROOT / f"configs/training/phase-{name}.yaml")
        assert phase.name == name and phase.patience == patience


def test_phase_arguments_keep_batch_and_data_order_contract() -> None:
    common = CommonTrainingConfig.from_yaml(ROOT / "configs/training/common.yaml")
    phase = load_phase_spec(ROOT / "configs/training/phase-c.yaml")

    args = common.to_ultralytics_args(phase, project=ROOT / "artifacts", name="probe")

    assert args["batch"] == args["nbs"] == 16
    assert args["seed"] == 0 and args["deterministic"] is True
    assert args["epochs"] == 55 and args["patience"] == 8 and args["cos_lr"] is True


def test_phase_c_recovery_supports_microbatch_eight_accumulate_two() -> None:
    common = CommonTrainingConfig.from_yaml(ROOT / "configs/training/common.yaml")
    recovery = replace(common, batch=8, nbs=16, workers=6, gradient_accumulation=True)
    phase = load_phase_spec(ROOT / "configs/training/phase-c.yaml")

    args = recovery.to_ultralytics_args(phase, project=ROOT / "artifacts", name="recovery-probe")

    assert args["batch"] == 8 and args["nbs"] == 16
    assert "gradient_accumulation" not in args
