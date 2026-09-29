#!/usr/bin/env python3
"""以獨立程序依序驗證 final bundle 中的 Bit-True 候選。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bundle import BUNDLE_ROOT, atomic_json, load_models

DEFAULT_OUTPUT = BUNDLE_ROOT / "outputs"
GIB = 1 << 30


def available_ram_bytes() -> int:
    """讀取 Linux MemAvailable。"""

    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo 缺少 MemAvailable")


def free_vram_bytes() -> int:
    """透過 nvidia-smi 取得 GPU 0 可用 VRAM。"""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.splitlines()[0].strip()) << 20


def selected_models(requested: list[str] | None) -> list[dict[str, Any]]:
    """依 registry 順序選出候選。"""

    records = load_models()
    if not requested:
        return records
    lookup = {record["id"]: record for record in records}
    unknown = sorted(set(requested) - set(lookup))
    if unknown:
        raise ValueError(f"未知 model id：{unknown}")
    return [lookup[item] for item in requested]


def run_worker(args: argparse.Namespace) -> int:
    """在單一 child process 執行一次 validation。"""

    checkpoint = BUNDLE_ROOT / args.checkpoint
    if args.dataset == "coco2017":
        from achitechure_1.evaluation import validate_bittrue

        validate_bittrue(
            checkpoint=checkpoint,
            data=args.data,
            run_dir=args.run_dir,
            imgsz=640,
            batch=args.batch,
            device="0",
            workers=args.workers,
        )
    else:
        from achitechure_1.ball_bat_evaluation import validate_ball_bat_checkpoint

        validate_ball_bat_checkpoint(
            checkpoint=checkpoint,
            data=args.data,
            run_dir=args.run_dir,
            imgsz=640,
            batch=args.batch,
            device="0",
            workers=args.workers,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("coco2017", "bbt5", "all"), default="all")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, choices=range(0, 9), default=6)
    parser.add_argument("--minimum-ram-gib", type=float, default=0.5)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=4.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--data", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)

    datasets = [args.dataset] if args.dataset != "all" else ["coco2017", "bbt5"]
    data_paths = {
        "coco2017": BUNDLE_ROOT / "configs/datasets/coco2017.yaml",
        "bbt5": BUNDLE_ROOT / "configs/datasets/bbt5-coco80.yaml",
    }
    state: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "jobs": [],
    }
    state_path = BUNDLE_ROOT / "queue-state.json"
    atomic_json(state_path, state)
    for record in selected_models(args.models):
        for dataset in datasets:
            run_dir = args.output.resolve() / str(record["id"]) / dataset
            metrics = run_dir / "metrics.json"
            if metrics.is_file():
                state["jobs"].append(
                    {"model": record["id"], "dataset": dataset, "status": "skipped-existing"}
                )
                atomic_json(state_path, state)
                continue
            ram = available_ram_bytes()
            vram = free_vram_bytes()
            if ram < int(args.minimum_ram_gib * GIB):
                raise RuntimeError(f"可用 RAM {ram / GIB:.2f} GiB 低於安全門檻")
            if vram < int(args.minimum_free_vram_gib * GIB):
                raise RuntimeError(f"可用 VRAM {vram / GIB:.2f} GiB 低於安全門檻")
            job = {
                "model": record["id"],
                "dataset": dataset,
                "status": "running",
                "available_ram_bytes": ram,
                "free_vram_bytes": vram,
            }
            state["jobs"].append(job)
            atomic_json(state_path, state)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--dataset",
                dataset,
                "--checkpoint",
                str(record["bittrue"]),
                "--data",
                str(data_paths[dataset]),
                "--run-dir",
                str(run_dir),
                "--batch",
                str(args.batch),
                "--workers",
                str(args.workers),
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(BUNDLE_ROOT / "code")
            subprocess.run(command, check=True, env=environment)
            job["status"] = "completed"
            job["metrics"] = str(metrics)
            atomic_json(state_path, state)
            gc.collect()
    state["status"] = "completed"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
