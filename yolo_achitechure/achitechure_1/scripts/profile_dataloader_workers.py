"""量測實際 COCO DataLoader 的 workers 吞吐與主機記憶體占用。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil
import torch
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from achitechure_1.config import CommonTrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _memory_available_bytes() -> int:
    return int(psutil.virtual_memory().available)


def _shared_memory_used_bytes() -> int:
    usage = shutil.disk_usage("/dev/shm")
    return int(usage.total - usage.free)


def _process_tree_memory() -> dict[str, int]:
    process = psutil.Process()
    processes = [process, *process.children(recursive=True)]
    totals = {"processes": 0, "rss_bytes": 0, "uss_bytes": 0, "pss_bytes": 0}
    for item in processes:
        try:
            memory = item.memory_full_info()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        totals["processes"] += 1
        totals["rss_bytes"] += int(memory.rss)
        totals["uss_bytes"] += int(memory.uss)
        totals["pss_bytes"] += int(memory.pss)
    return totals


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _loader_config(common: CommonTrainingConfig, fraction: float):
    cfg = get_cfg(DEFAULT_CFG)
    for key, value in asdict(common).items():
        if key != "gradient_accumulation" and hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.task = "detect"
    cfg.fraction = fraction
    return cfg


def profile(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    common = CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml")
    cfg = _loader_config(common, args.fraction)
    data = check_det_dataset(str((root / common.data).resolve()))

    baseline = {
        "system_available_bytes": _memory_available_bytes(),
        "shared_memory_used_bytes": _shared_memory_used_bytes(),
        "process_tree": _process_tree_memory(),
    }
    dataset = build_yolo_dataset(
        cfg,
        data["train"],
        batch=args.batch,
        data=data,
        mode="train",
        stride=32,
        fraction=args.fraction,
    )
    dataset_ready = {
        "system_available_bytes": _memory_available_bytes(),
        "shared_memory_used_bytes": _shared_memory_used_bytes(),
        "process_tree": _process_tree_memory(),
    }
    loader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=True,
        rank=-1,
        pin_memory=True,
    )
    iterator = iter(loader)
    batch = next(iterator)
    time.sleep(args.prefetch_wait)

    start = time.perf_counter()
    for _ in range(args.measured_batches):
        batch = next(iterator)
    elapsed = time.perf_counter() - start
    time.sleep(args.prefetch_wait)

    measured = {
        "system_available_bytes": _memory_available_bytes(),
        "shared_memory_used_bytes": _shared_memory_used_bytes(),
        "process_tree": _process_tree_memory(),
    }
    payload = {
        "batch": args.batch,
        "workers_requested": args.workers,
        "workers_actual": int(loader.num_workers),
        "prefetch_factor": loader.prefetch_factor,
        "pin_memory": bool(loader.pin_memory),
        "fraction": args.fraction,
        "dataset_images": len(dataset),
        "measured_batches": args.measured_batches,
        "batch_tensor_bytes": _tensor_bytes(batch),
        "elapsed_seconds": elapsed,
        "milliseconds_per_batch": elapsed * 1000.0 / args.measured_batches,
        "images_per_second": args.batch * args.measured_batches / elapsed,
        "baseline": baseline,
        "dataset_ready": dataset_ready,
        "measured": measured,
        "process_pss_increase_from_dataset_ready_bytes": (
            measured["process_tree"]["pss_bytes"] - dataset_ready["process_tree"]["pss_bytes"]
        ),
        "process_uss_increase_from_dataset_ready_bytes": (
            measured["process_tree"]["uss_bytes"] - dataset_ready["process_tree"]["uss_bytes"]
        ),
        "system_available_decrease_from_dataset_ready_bytes": (
            dataset_ready["system_available_bytes"] - measured["system_available_bytes"]
        ),
        "shared_memory_increase_from_dataset_ready_bytes": (
            measured["shared_memory_used_bytes"] - dataset_ready["shared_memory_used_bytes"]
        ),
    }
    loader.close()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--measured-batches", type=int, default=20)
    parser.add_argument("--prefetch-wait", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 0 or args.batch < 1 or args.measured_batches < 1:
        raise ValueError("workers 不得為負數，batch 與 measured-batches 必須為正數")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction 必須介於 0 與 1 之間")
    payload = profile(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"workers={payload['workers_actual']}：{payload['images_per_second']:.1f} images/s，"
        f"PSS 增量 {payload['process_pss_increase_from_dataset_ready_bytes'] / (1 << 20):.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
