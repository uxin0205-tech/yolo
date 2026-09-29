#!/usr/bin/env python3
"""Profile every Full35 activation site on the complete training splits."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activation_lab.training.full35 import (
    Full35ActivationExperiment,
    load_full35_manifest,
    uniform_full35_policy,
)

DEFAULT_RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"
QUANTILES = (0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999)


@dataclass
class _RunningStats:
    bins: int
    minimum: float
    maximum: float
    count: int = 0
    total: torch.Tensor | None = None
    squared_total: torch.Tensor | None = None
    observed_minimum: torch.Tensor | None = None
    observed_maximum: torch.Tensor | None = None
    negative: torch.Tensor | None = None
    zero: torch.Tensor | None = None
    tail: torch.Tensor | None = None
    nonfinite: torch.Tensor | None = None
    histogram: torch.Tensor | None = None

    def update(self, value: torch.Tensor) -> None:
        data = value.detach().float()
        device = data.device
        if self.total is None:
            self.total = torch.zeros((), dtype=torch.float64, device=device)
            self.squared_total = torch.zeros((), dtype=torch.float64, device=device)
            self.observed_minimum = torch.full((), float("inf"), device=device)
            self.observed_maximum = torch.full((), float("-inf"), device=device)
            self.negative = torch.zeros((), dtype=torch.int64, device=device)
            self.zero = torch.zeros((), dtype=torch.int64, device=device)
            self.tail = torch.zeros((), dtype=torch.int64, device=device)
            self.nonfinite = torch.zeros((), dtype=torch.int64, device=device)
            self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=device)
        finite = torch.isfinite(data)
        safe = torch.nan_to_num(
            data,
            nan=0.0,
            posinf=self.maximum,
            neginf=self.minimum,
        )
        self.count += safe.numel()
        self.total += safe.sum(dtype=torch.float64)
        self.squared_total += safe.square().sum(dtype=torch.float64)
        self.observed_minimum.copy_(torch.minimum(self.observed_minimum, safe.min()))
        self.observed_maximum.copy_(torch.maximum(self.observed_maximum, safe.max()))
        self.negative += (safe < 0).sum()
        self.zero += (safe == 0).sum()
        self.tail += (safe.abs() >= 8.0).sum()
        self.nonfinite += (~finite).sum()
        self.histogram += torch.histc(
            safe.clamp(self.minimum, self.maximum),
            bins=self.bins,
            min=self.minimum,
            max=self.maximum,
        ).to(dtype=torch.float64)

    def _quantiles(self, histogram: torch.Tensor) -> dict[str, float]:
        cumulative = histogram.cumsum(0)
        total = float(cumulative[-1])
        width = (self.maximum - self.minimum) / self.bins
        result: dict[str, float] = {}
        for quantile in QUANTILES:
            target = torch.tensor(quantile * total, dtype=cumulative.dtype)
            index = int(torch.searchsorted(cumulative, target).clamp(max=self.bins - 1))
            result[f"p{str(quantile).replace('.', '')}"] = (
                self.minimum + (index + 0.5) * width
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        if self.total is None or self.histogram is None or self.count < 1:
            raise RuntimeError("activation site received no tensors")
        total = float(self.total)
        mean = total / self.count
        variance = max(float(self.squared_total) / self.count - mean * mean, 0.0)
        histogram = self.histogram.cpu()
        return {
            "count": self.count,
            "mean": mean,
            "std": variance**0.5,
            "min": float(self.observed_minimum),
            "max": float(self.observed_maximum),
            "negative_fraction": float(self.negative) / self.count,
            "zero_fraction": float(self.zero) / self.count,
            "tail_abs_ge_8_fraction": float(self.tail) / self.count,
            "nonfinite_count": int(self.nonfinite),
            "histogram": {
                "minimum": self.minimum,
                "maximum": self.maximum,
                "bins": self.bins,
                "counts": [int(value) for value in histogram.tolist()],
            },
            **self._quantiles(histogram),
        }


class _Observer:
    def __init__(self, *, bins: int, minimum: float, maximum: float) -> None:
        self.bins = bins
        self.minimum = minimum
        self.maximum = maximum
        self.states: dict[str, _RunningStats] = {}

    def record(self, path: str, value: torch.Tensor) -> None:
        state = self.states.setdefault(
            path,
            _RunningStats(self.bins, self.minimum, self.maximum),
        )
        state.update(value)


class _ActivationProbe(nn.Module):
    def __init__(self, path: str, activation: nn.Module, observer: _Observer) -> None:
        super().__init__()
        self.path = path
        self.activation = activation
        self.observer = observer

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.observer.record(self.path, value)
        return self.activation(value)


def _set_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = replacement
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = replacement
    else:
        setattr(parent, child_name, replacement)


def _install_probes(
    model: nn.Module,
    paths: tuple[str, ...],
    observer: _Observer,
) -> None:
    for path in paths:
        activation = model.get_submodule(path)
        if not isinstance(activation, nn.SiLU):
            raise TypeError(f"profiling source is not SiLU: {path}")
        _set_submodule(model, path, _ActivationProbe(path, activation, observer))


def _training_forward_graph(model: nn.Module) -> nn.Module:
    """Expose auxiliary heads without updating frozen Full35 BN statistics."""

    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return model


def _profile_step(
    model: nn.Module,
    *,
    task: str,
    batch: dict[str, Any],
    loss_router: Any | None,
) -> None:
    if task == "pose":
        if loss_router is None:
            raise RuntimeError("pose profiling requires the formal loss router")
        loss_router.loss_for(task, batch)
        return
    model(batch["img"], task=task)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full35 完整 training split profiling")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--task", choices=("both", "detect", "pose"), default="both")
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--hist-min", type=float, default=-16.0)
    parser.add_argument("--hist-max", type=float, default=16.0)
    parser.add_argument("--name", default="profile-full-train-seed1")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新計算指定 task 並覆寫可重建的 profile report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bins < 32 or args.hist_min >= args.hist_max:
        raise ValueError("histogram configuration is invalid")
    experiment = Full35ActivationExperiment.from_yaml(args.recipe)
    manifest = load_full35_manifest(experiment.config)
    loaded = experiment.load_policy_model(manifest, uniform_full35_policy("silu"))
    modules = {
        name: importlib.import_module(f"yolo_combine.{name}")
        for name in ("data", "joint_data", "joint_loss")
    }
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    output_root = experiment.config.run_root / "profiling" / args.name
    pose_view = modules["data"].prepare_bbt5_view(
        experiment.config.bbat5_registry,
        output_root / "datasets/bbat5-v1-runtime",
    )
    tasks = ("detect", "pose") if args.task == "both" else (args.task,)
    loader_settings = modules["joint_data"].TaskLoaderSettings
    build_loader = modules["joint_data"].build_task_loader
    site_regions = {site.module_path: site.region for site in manifest.sites}
    task_reports: dict[str, Any] = {}
    for task in tasks:
        target = output_root / f"{task}-activation-stats.json"
        if target.is_file() and not args.overwrite:
            task_reports[task] = json.loads(target.read_text(encoding="utf-8"))
            print(f"reused={target}", flush=True)
            continue
        observer = _Observer(
            bins=args.bins,
            minimum=args.hist_min,
            maximum=args.hist_max,
        )
        clean = _training_forward_graph(
            experiment.load_policy_model(
                manifest, uniform_full35_policy("silu")
            ).model.to(device)
        )
        _install_probes(clean, tuple(site_regions), observer)
        if task == "detect":
            settings = loader_settings.for_detect(
                batch_size=32,
                workers=loaded.joint_config.detect_workers,
                imgsz=loaded.joint_config.imgsz,
                fraction=1.0,
                seed=1,
            )
            data_yaml = experiment.config.coco_detect
        else:
            settings = loader_settings.for_pose(
                batch_size=16,
                workers=loaded.joint_config.pose_workers,
                imgsz=loaded.joint_config.imgsz,
                fraction=1.0,
                seed=1,
            )
            data_yaml = pose_view.yaml
        prepared = build_loader(
            clean,
            data_yaml=data_yaml,
            settings=settings,
            device=device,
            registry=experiment.config.bbat5_registry,
        )
        loss_router = (
            modules["joint_loss"].NativeTaskLossRouter(
                clean,
                epochs=1,
                imgsz=loaded.joint_config.imgsz,
            )
            if task == "pose"
            else None
        )
        images_seen = 0
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(prepared.loader, start=1):
                batch = prepared.preprocess(raw_batch)
                images = batch["img"]
                images_seen += int(images.shape[0])
                _profile_step(
                    clean,
                    task=task,
                    batch=batch,
                    loss_router=loss_router,
                )
                if batch_index % 100 == 0:
                    print(
                        f"task={task} batches={batch_index} images={images_seen}",
                        flush=True,
                    )
        sites = {
            path: {"region": site_regions[path], **state.to_dict()}
            for path, state in sorted(observer.states.items())
        }
        other_head = "pose_" if task == "detect" else "detect_"
        expected_paths = {
            path
            for path, region in site_regions.items()
            if not region.startswith(other_head)
        }
        missing = sorted(expected_paths - set(sites))
        inactive_other_head_sites = sorted(set(site_regions) - expected_paths)
        report = {
            "schema_version": 1,
            "checkpoint": str(experiment.config.checkpoint),
            "checkpoint_sha256": experiment.config.checkpoint_sha256,
            "joint_config": str(experiment.config.joint_config),
            "task": task,
            "data_yaml": str(data_yaml),
            "split": "train",
            "fraction": 1.0,
            "resampling": False,
            "seed": 1,
            "dataset_images": len(prepared.dataset),
            "images_seen": images_seen,
            "manifest_site_count": len(manifest.sites),
            "profiling_site_count": len(site_regions),
            "loss_path_profiled": task == "pose",
            "inactive_other_head_sites": inactive_other_head_sites,
            "observed_site_count": len(sites),
            "missing_sites": missing,
            "sites": sites,
        }
        _write(target, report)
        task_reports[task] = report
        print(f"completed={target}", flush=True)
    for existing_task in ("detect", "pose"):
        existing = output_root / f"{existing_task}-activation-stats.json"
        if existing_task not in task_reports and existing.is_file():
            task_reports[existing_task] = json.loads(
                existing.read_text(encoding="utf-8")
            )
    _write(
        output_root / "profile-summary.json",
        {
            "schema_version": 1,
            "checkpoint_sha256": experiment.config.checkpoint_sha256,
            "joint_config": str(experiment.config.joint_config),
            "tasks": {
                task: {
                    "data_yaml": report["data_yaml"],
                    "dataset_images": report["dataset_images"],
                    "images_seen": report["images_seen"],
                    "observed_site_count": report["observed_site_count"],
                    "missing_sites": report["missing_sites"],
                }
                for task, report in task_reports.items()
            },
        },
    )
    print(f"summary={output_root / 'profile-summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
