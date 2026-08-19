"""Launch the RTX 4080 A1 recovery run with fewer data workers."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from achitechure_1.config import CommonTrainingConfig, load_phase_spec
from achitechure_1.training import launch_phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if "rtx4080super-batch16-workers4" not in args.run_id.lower():
        raise ValueError("recovery run-id must disclose rtx4080super-batch16-workers4")

    root = Path(__file__).resolve().parents[1]
    common = replace(
        CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml"),
        batch=16,
        nbs=16,
        workers=4,
    )
    launch_phase(
        project_root=root,
        weights=root / "artifacts/prepared/a1-full35.pt",
        masf_variant="full35",
        attention_config=root / "configs/attention/float-pwl-final.yaml",
        bittrue_config=root / "configs/attention/bittrue-pwl-final.yaml",
        phase=load_phase_spec(root / "configs/training/phase-a1.yaml"),
        common=common,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
