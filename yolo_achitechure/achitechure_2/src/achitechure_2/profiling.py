"""Full35 Float20 候選的可重跑結構成本與 GPU inference profile。"""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .candidate import build_candidate
from .config import SPEC_PATH, SPEC_VERSION, file_sha256
from .full35_adapter import Full35Release
from .screen_training import ScreenRunConfig


@dataclass(frozen=True)
class TaskProfile:
    """單一 task route 的固定 batch=1 Float latency 與 VRAM。"""

    task: str
    gflops: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    baseline_allocated_mib: float
    peak_allocated_mib: float
    peak_reserved_mib: float


@dataclass(frozen=True)
class CandidateCostProfile:
    """一個 resolved candidate 的結構、checkpoint 與三路 inference 成本。"""

    candidate: str
    params: int
    trainable_params: int
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_source: str
    tasks: dict[str, TaskProfile]


class _TaskForward(nn.Module):
    def __init__(self, model: nn.Module, task: str) -> None:
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, images: torch.Tensor) -> Any:
        return self.model(images, task=self.task)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency samples 不得為空")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _gflops(model: nn.Module, *, task: str, imgsz: int) -> float:
    try:
        import thop
    except ImportError as error:
        raise RuntimeError("正式 GFLOPs profile 需要 ultralytics-thop") from error

    parameter = next(model.parameters())
    images = torch.zeros(
        1,
        3,
        imgsz,
        imgsz,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    wrapper = _TaskForward(model, task).eval()
    with torch.inference_mode():
        macs, _ = thop.profile(wrapper, inputs=(images,), verbose=False)
    gflops = float(macs) * 2.0 / 1e9
    if not math.isfinite(gflops) or gflops <= 0:
        raise RuntimeError(f"{task} GFLOPs 無效：{gflops}")
    return gflops

def _prepare_profile_model(
    model: nn.Module,
    *,
    device: torch.device,
    imgsz: int,
) -> tuple[nn.Module, dict[str, float]]:
    """先搬到最終device，再建立head動態cache與GFLOPs。"""

    model = model.eval().to(device)
    task_flops = {
        task: _gflops(model, task=task, imgsz=imgsz)
        for task in ("detect", "pose", "both")
    }
    return model, task_flops


def _load_inference_checkpoint(model: nn.Module, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("checkpoint_kind") != "inference_only"
        or payload.get("source") != "ema"
    ):
        raise ValueError(f"不支援的 inference checkpoint：{checkpoint}")
    contract = model.contract()
    if payload.get("contract") != contract:
        raise ValueError(f"inference checkpoint contract 漂移：{checkpoint}")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise TypeError("inference checkpoint 缺少 state_dict")
    model.load_state_dict(state, strict=True)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("inference checkpoint metadata 必須是 mapping")
    return payload


def _profile_task(
    model: nn.Module,
    images: torch.Tensor,
    *,
    task: str,
    gflops: float,
    device: torch.device,
    amp: bool,
    warmup: int,
    iterations: int,
) -> TaskProfile:
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            model(images, task=task)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = float(torch.cuda.memory_allocated(device)) / 2**20
    timings: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            model(images, task=task)
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"{task} latency sample 無效：{elapsed}")
        timings.append(elapsed)
    return TaskProfile(
        task=task,
        gflops=gflops,
        latency_mean_ms=statistics.fmean(timings),
        latency_median_ms=statistics.median(timings),
        latency_p95_ms=_percentile(timings, 0.95),
        baseline_allocated_mib=baseline,
        peak_allocated_mib=float(torch.cuda.max_memory_allocated(device)) / 2**20,
        peak_reserved_mib=float(torch.cuda.max_memory_reserved(device)) / 2**20,
    )


def profile_float20_candidates(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    warmup: int = 25,
    iterations: int = 100,
) -> dict[str, Any]:
    """嚴格載入每個 best-joint EMA，以同一 GPU 設定 profile 三種 route。"""

    if warmup < 1 or iterations < 2:
        raise ValueError("profile warmup 必須 >=1 且 iterations 必須 >=2")
    config = ScreenRunConfig.load(config_path)
    matrix_path = config.run_root / "matrix-complete.json"
    if not matrix_path.is_file():
        raise FileNotFoundError("完整 C0～C3 matrix 尚未完成，拒絕 profile")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("status") != "completed_screening_matrix":
        raise ValueError("matrix-complete status 漂移")
    if not torch.cuda.is_available():
        raise RuntimeError("正式 Float20 profile 需要 CUDA")
    device_index = int(config.device)
    if device_index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device_index} 不存在")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    release = Full35Release(config.full35_root)
    parent = release.load_parent()
    profiles: list[CandidateCostProfile] = []
    for candidate in config.candidates:
        resolved = release.resolved_candidate(candidate)
        model, build = build_candidate(parent.model, resolved, seed=config.seed)
        checkpoint = (
            config.run_root
            / f"{candidate.lower()}-control-seed{config.seed}"
            / "inference/best-joint-screening.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = _load_inference_checkpoint(model, checkpoint)
        metadata = payload["metadata"]
        if metadata.get("candidate") != candidate:
            raise ValueError(f"{candidate} checkpoint metadata candidate 漂移")
        if build.resolved_id != candidate or not build.model_contract_unchanged:
            raise ValueError(f"{candidate} build contract 漂移")

        model, task_flops = _prepare_profile_model(
            model,
            device=device,
            imgsz=config.imgsz,
        )
        params = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        images = torch.zeros(1, 3, config.imgsz, config.imgsz, device=device)
        with torch.inference_mode():
            tasks = {
                task: _profile_task(
                    model,
                    images,
                    task=task,
                    gflops=task_flops[task],
                    device=device,
                    amp=config.amp,
                    warmup=warmup,
                    iterations=iterations,
                )
                for task in ("detect", "pose", "both")
            }
        profiles.append(
            CandidateCostProfile(
                candidate=candidate,
                params=params,
                trainable_params=trainable,
                checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=file_sha256(checkpoint),
                checkpoint_source=str(payload["source"]),
                tasks=tasks,
            )
        )
        del images, model, payload
        gc.collect()
        torch.cuda.empty_cache()

    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else config.run_root / "profiles/cost-profiles.json"
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "scope": "Float20 best-joint EMA PyTorch inference profile",
        "screening_only": True,
        "simulation_only": False,
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "training_yaml": str(config.path),
        "training_yaml_sha256": file_sha256(config.path),
        "parent_checkpoint_sha256": file_sha256(release.checkpoint),
        "device": {
            "index": device_index,
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "amp": config.amp,
            "dtype": "float16_autocast" if config.amp else "float32",
            "batch": 1,
            "imgsz": config.imgsz,
            "warmup": warmup,
            "iterations": iterations,
            "synchronize_each_iteration": True,
        },
        "profiles": [
            {
                **{
                    key: value
                    for key, value in asdict(profile).items()
                    if key != "tasks"
                },
                "tasks": {
                    task: asdict(task_profile)
                    for task, task_profile in profile.tasks.items()
                },
            }
            for profile in profiles
        ],
    }
    _atomic_json(destination, report)
    report["output"] = str(destination)
    return report
