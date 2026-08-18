from __future__ import annotations

from pathlib import Path

from achitechure_1.config import CommonTrainingConfig, load_phase_spec

ROOT = Path(__file__).resolve().parents[1]


def test_training_configs_disable_accumulation_and_match_fixed_phases() -> None:
    common = CommonTrainingConfig.from_yaml(ROOT / "configs/training/common.yaml")

    assert common.batch == 16 and common.nbs == 16
    assert common.optimizer == "MuSGD"
    assert common.imgsz == 640 and common.amp is True and common.workers == 8
    for name in ("a1", "a2", "b", "c"):
        assert load_phase_spec(ROOT / f"configs/training/phase-{name}.yaml").name == name


def test_phase_arguments_keep_batch_and_data_order_contract() -> None:
    common = CommonTrainingConfig.from_yaml(ROOT / "configs/training/common.yaml")
    phase = load_phase_spec(ROOT / "configs/training/phase-c.yaml")

    args = common.to_ultralytics_args(phase, project=ROOT / "artifacts", name="probe")

    assert args["batch"] == args["nbs"] == 16
    assert args["seed"] == 0 and args["deterministic"] is True
    assert args["epochs"] == 55 and args["patience"] == 5 and args["cos_lr"] is True
