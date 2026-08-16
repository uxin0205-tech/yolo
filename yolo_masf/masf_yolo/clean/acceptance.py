"""CUDA batch-16 acceptance gate for every unique Clean architecture."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from ..artifacts.io import atomic_write_json
from ..training.preflight import run_optimizer_step
from .builder import build_clean_model
from .contracts import CLEAN_EXPERIMENTS, load_clean_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/clean/clean_ablation.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("acceptance requires exactly one visible CUDA device")
    config = load_clean_config(args.config)
    config.verify_initializer()
    device = torch.device("cuda:0")
    batch_size = int(config.values["training"]["batch"])
    imgsz = int(config.values["training"]["imgsz"])
    batch = {
        "img": torch.rand(batch_size, 3, imgsz, imgsz),
        "batch_idx": torch.arange(batch_size),
        "cls": torch.zeros(batch_size, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]).repeat(batch_size, 1),
    }
    seen: set[tuple[str, str | None]] = set()
    results: list[dict] = []
    for experiment, spec in CLEAN_EXPERIMENTS.items():
        key = (spec.family, spec.variant)
        if key in seen:
            continue
        seen.add(key)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_clean_model(experiment, config.initializer_path)
        loss = run_optimizer_step(model, batch, device=device, amp=True)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        peak = torch.cuda.max_memory_allocated()
        del model
        torch.cuda.empty_cache()
        reloaded = build_clean_model(experiment, config.initializer_path)
        reloaded.load_state_dict(state, strict=True)
        results.append({
            "experiment": experiment,
            "family": spec.family,
            "variant": spec.variant,
            "batch": batch_size,
            "imgsz": imgsz,
            "amp": True,
            "finite_loss": loss,
            "peak_memory_bytes": peak,
            "strict_reload": True,
        })
        del reloaded, state
        gc.collect()
    payload = {
        "ok": True,
        "training_started": False,
        "device_name": torch.cuda.get_device_name(),
        "architectures": results,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
