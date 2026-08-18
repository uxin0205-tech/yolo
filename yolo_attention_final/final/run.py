"""One entrypoint for checking, predicting, and reproducing the queued training recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from training_recipe import DEFAULT_QUEUE_ROOT, print_recipe, queue_status, run_training
from ultralytics import YOLO

from yolo_attention.attention import HardwareFriendlyAttention
from yolo_attention.config import NormalizationKind

# Block 1: the single delivered checkpoint and fixed model contract.
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "pwl-final-best.pt"
EXPECTED_SHA256 = "c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123"
ATTENTION_SITES = ("model.10.m.0.attn", "model.22.m.0.1.attn")


# Block 2: load the checkpoint and fail closed on any contract mismatch.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_check() -> tuple[YOLO, dict[str, object]]:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    digest = sha256_file(WEIGHTS)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: {digest}")

    yolo = YOLO(str(WEIGHTS))
    model = yolo.model
    yaml = model.yaml
    sites = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, HardwareFriendlyAttention)
    }
    if yaml.get("yaml_file") != "yolo26m.yaml" or yaml.get("scale") != "m" or yaml.get("nc") != 80:
        raise RuntimeError("checkpoint is not official YOLO26m scale=m with 80 classes")
    if set(sites) != set(ATTENTION_SITES):
        raise RuntimeError(f"unexpected Attention sites: {sorted(sites)}")

    for name, module in sites.items():
        normalizer = module.normalize
        if module.config.normalization is not NormalizationKind.BIT_TRUE_PWL:
            raise RuntimeError(f"{name} is not Bit-True PWL")
        if module.score.use_ste:
            raise RuntimeError(f"{name} unexpectedly enables STE")
        if normalizer.score_floor != -10.0 or normalizer.segments != 20:
            raise RuntimeError(f"{name} has the wrong PWL range or segment count")
        if normalizer.endpoint_table.numel() != 21 or normalizer.endpoint_storage_bits != 336:
            raise RuntimeError(f"{name} has the wrong endpoint table")

    report = {
        "weights": str(WEIGHTS),
        "sha256": digest,
        "model": "yolo26m.yaml",
        "scale": "m",
        "classes": 80,
        "attention_sites": sorted(sites),
        "normalization": "Bit-True PWL Q8.8/UQ1.15, [-10, 0], 20 segments",
    }
    return yolo, report


# Block 3: expose prediction and the exact queue-based training workflow.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("check", help="verify SHA-256 and the complete model contract")
    commands.add_parser("recipe", help="print the exact training stages and learning rates")

    predict = commands.add_parser("predict", help="run Ultralytics prediction")
    predict.add_argument("source", help="image, directory, video, or stream source")
    predict.add_argument("--device", default=None, help="for example cpu or 0")
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--save", action="store_true")

    train = commands.add_parser("train", help="preview the queue; --execute starts or resumes it")
    train.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    train.add_argument("--execute", action="store_true")

    status = commands.add_parser("status", help="show a reproduction queue status")
    status.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    return parser


# Block 4: keep every user operation behind this one small command surface.
def main() -> int:
    args = build_parser().parse_args()
    command = args.command or "check"
    if command == "check":
        _, report = load_and_check()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if command == "recipe":
        print_recipe()
        return 0
    if command == "predict":
        yolo, report = load_and_check()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        yolo.predict(
            source=args.source,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            save=args.save,
        )
        return 0
    if command == "train":
        return run_training(args.queue_root, execute=args.execute)
    if command == "status":
        return queue_status(args.queue_root)
    raise AssertionError(f"unsupported command: {command}")


if __name__ == "__main__":
    sys.exit(main())
