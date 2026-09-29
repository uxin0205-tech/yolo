"""Fresh-process conversion from native Ultralytics best EMA to canonical state dict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from masf_yolo.contracts import sha256_file
from masf_yolo.variants import get_variant

from .checkpoints import save_canonical_checkpoint
from .io import atomic_write_json


def build_finalize_command(
    *,
    python: Path,
    source: Path,
    checkpoint: Path,
    variant_id: str,
    data_hash: str,
    config_hash: str,
    environment_hash: str,
    output: Path,
) -> list[str]:
    return [
        str(python),
        "-m",
        "masf_yolo.artifacts.finalize",
        "--source",
        str(source),
        "--checkpoint",
        str(checkpoint),
        "--variant",
        variant_id,
        "--data-hash",
        data_hash,
        "--config-hash",
        config_hash,
        "--environment-hash",
        environment_hash,
        "--output",
        str(output),
    ]


def finalize_native_checkpoint(
    source: Path,
    checkpoint: Path,
    *,
    variant_id: str,
    data_hash: str,
    config_hash: str,
    environment_hash: str,
) -> dict[str, str]:
    source = source.resolve()
    checkpoint = checkpoint.resolve()
    variant = get_variant(variant_id)
    model = YOLO(str(source), task="detect").model.float().cpu()
    actual_variant = getattr(model, "masf_variant", None)
    if actual_variant not in (None, variant_id):
        raise ValueError(f"native checkpoint variant mismatch: {actual_variant} != {variant_id}")
    model.masf_variant = variant_id
    model.masf_variant_hash = variant.config_hash
    manifest = save_canonical_checkpoint(
        model,
        checkpoint,
        variant,
        data_hash=data_hash,
        config_hash=config_hash,
        environment_hash=environment_hash,
    )
    return {
        "variant": variant_id,
        "source": str(source),
        "source_hash": sha256_file(source),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": manifest.checkpoint_hash,
        "state_dict_hash": manifest.state_dict_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data-hash", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--environment-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_native_checkpoint(
        args.source,
        args.checkpoint,
        variant_id=args.variant,
        data_hash=args.data_hash,
        config_hash=args.config_hash,
        environment_hash=args.environment_hash,
    )
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
