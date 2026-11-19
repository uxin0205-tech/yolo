from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from achitechure_1.config import CommonTrainingConfig, load_phase_spec, training_config_for_phase
from achitechure_1.preflight import EXPECTED_TORCH_VERSION, EXPECTED_ULTRALYTICS_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_training_configs_use_batch_sixteen_and_match_fixed_phases() -> None:
    common = CommonTrainingConfig.from_yaml(ROOT / "configs/training/common.yaml")

    assert common.batch == 16 and common.nbs == 16
    assert common.gradient_accumulation is False
    assert common.optimizer == "MuSGD"
    assert common.imgsz == 640 and common.amp is True and common.workers == 6
    assert common.fraction == 0.3
    expected_patience = {"a1": 100, "a2": 4, "b": 4, "c": 9}
    for name, patience in expected_patience.items():
        phase = load_phase_spec(ROOT / f"configs/training/phase-{name}.yaml")
        assert phase.name == name and phase.patience == patience


def test_formal_runtime_versions_are_pinned() -> None:
    requirements = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")

    assert f"torch=={EXPECTED_TORCH_VERSION}" in requirements
    assert f"ultralytics=={EXPECTED_ULTRALYTICS_VERSION}" in requirements


def test_phase_c_arguments_apply_batch_eight_accumulate_two_contract() -> None:
    common = CommonTrainingConfig.from_yaml(ROOT / "configs/training/common.yaml")
    phase = load_phase_spec(ROOT / "configs/training/phase-c.yaml")
    common = training_config_for_phase(common, phase.name)

    args = common.to_ultralytics_args(phase, project=ROOT / "artifacts", name="probe")

    assert args["batch"] == 8 and args["nbs"] == 16
    assert args["seed"] == 0 and args["deterministic"] is True
    assert args["workers"] == 6 and args["fraction"] == 0.3
    assert args["epochs"] == 70 and args["patience"] == 9 and args["cos_lr"] is True


def test_phase_c_recovery_supports_microbatch_eight_accumulate_two() -> None:
    common = CommonTrainingConfig.from_yaml(ROOT / "configs/training/common.yaml")
    recovery = replace(common, batch=8, nbs=16, workers=6, gradient_accumulation=True)
    phase = load_phase_spec(ROOT / "configs/training/phase-c.yaml")

    args = recovery.to_ultralytics_args(phase, project=ROOT / "artifacts", name="recovery-probe")

    assert args["batch"] == 8 and args["nbs"] == 16
    assert "gradient_accumulation" not in args
