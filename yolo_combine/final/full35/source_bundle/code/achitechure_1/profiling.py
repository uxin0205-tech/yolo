"""Comparable FP16 inference and real-loss smoke-training profiles."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.utils.torch_utils import get_flops
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model

from .model import inspect_yolo26_graph
from .phases import PHASES, apply_phase_scope, build_phase_optimizer, enforce_frozen_modules_eval


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_official_loss(model: Any) -> None:
    """Restore the full attribute-based training args stripped from saved checkpoints."""
    checkpoint_args = dict(model.args) if isinstance(model.args, Mapping) else vars(model.args)
    model.args = get_cfg({**DEFAULT_CFG_DICT, **checkpoint_args})
    model.criterion = None


def _masf_macs(model, imgsz: int) -> int:
    graph = inspect_yolo26_graph(model)
    masf = getattr(model.model[graph.p3_index], "p3_masf", None)
    if masf is None:
        return 0
    channels = getattr(masf, "context_channels", masf.channels)
    spatial = (imgsz // 8) ** 2
    return int((channels * 9 + channels * 25 + channels * channels) * spatial)


def profile_checkpoint(
    *,
    checkpoint: Path,
    output: Path,
    imgsz: int = 640,
    device_name: str = "0",
    warmup: int = 20,
    iterations: int = 100,
) -> Path:
    """Measure FP16 batch-1 p50 latency, Params, GFLOPs, and peak VRAM."""

    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    device = torch.device(f"cuda:{device_name}" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(checkpoint.resolve()))
    model = yolo.model.eval().to(device)
    inspect_yolo26_graph(model)
    fp16 = device.type == "cuda"
    model.half() if fp16 else model.float()
    sample = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=torch.float16 if fp16 else torch.float32)
    if fp16:
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        _sync(device)
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            _sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "precision": "fp16" if fp16 else "fp32",
        "batch": 1,
        "imgsz": imgsz,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gflops": float(get_flops(model, imgsz=imgsz)),
        "p3_masf_macs": _masf_macs(model, imgsz),
        "latency_ms": {
            "p50": statistics.median(timings),
            "mean": statistics.fmean(timings),
            "min": min(timings),
            "max": max(timings),
        },
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if fp16 else None,
        "gpu": torch.cuda.get_device_name(device) if fp16 else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def profile_training_step(
    *,
    checkpoint: Path,
    float_attention_config: Path,
    output: Path,
    batch: int = 16,
    imgsz: int = 640,
    device_name: str = "0",
    steps: int = 3,
    accumulate: int = 1,
    amp: bool = False,
) -> Path:
    """量測正式 YOLO loss，可選擇梯度累積與 AMP。"""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal training-step profile")
    if accumulate < 1:
        raise ValueError("accumulate must be positive")
    device = torch.device(f"cuda:{device_name}")
    model = YOLO(str(checkpoint.resolve())).model
    convert_yolo26_model(model, VariantConfig.from_yaml(float_attention_config))
    inspect_yolo26_graph(model)
    if not hasattr(model.model[inspect_yolo26_graph(model).p3_index], "p3_masf"):
        raise ValueError("training-step profile requires an A1/A2 MASF checkpoint")
    model.to(device).train()
    configure_official_loss(model)
    apply_phase_scope(model, "c")
    enforce_frozen_modules_eval(model)
    optimizer = build_phase_optimizer(model, PHASES["c"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    images = torch.rand(batch, 3, imgsz, imgsz, device=device)
    targets = {
        "img": images,
        "batch_idx": torch.arange(batch, device=device),
        "cls": torch.zeros(batch, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device).repeat(batch, 1),
    }
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        for _ in range(accumulate):
            with torch.amp.autocast("cuda", enabled=amp):
                loss, _ = model(targets)
            scaler.scale(loss.sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        _sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    payload = {
        "status": "passed",
        "checkpoint": str(checkpoint.resolve()),
        "batch": batch,
        "accumulate": accumulate,
        "effective_batch": batch * accumulate,
        "amp": amp,
        "imgsz": imgsz,
        "steps": steps,
        "train_step_ms": {"p50": statistics.median(timings), "mean": statistics.fmean(timings)},
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
