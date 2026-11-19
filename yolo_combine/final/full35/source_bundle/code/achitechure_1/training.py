"""Ultralytics trainer adapter for staged MASF fine-tuning."""

from __future__ import annotations

import copy
import csv
import ctypes
import gc
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model

from .checkpoint import build_training_model, file_sha256
from .config import CommonTrainingConfig
from .phases import (
    PhaseSpec,
    apply_phase_scope,
    assert_frozen_state_unchanged,
    build_phase_optimizer,
    enforce_frozen_modules_eval,
    snapshot_frozen_state,
)
from .runtime import nvidia_driver_version

MAP_SELECTION_METRIC = "metrics/mAP50-95(B)"
RESOURCE_CHECK_INTERVAL_BATCHES = 100
RUNTIME_MIN_AVAILABLE_RAM_BYTES = 3 << 29
RUNTIME_MIN_FREE_VRAM_BYTES = 1 << 30


def _memory_available_bytes() -> int:
    """讀取 kernel 納入可回收頁面後的可用記憶體估計。"""

    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo 未提供 MemAvailable")


def _process_memory_bytes() -> dict[str, int | None]:
    """不依賴額外套件，讀取目前 process 的 RSS 與 PSS。"""

    result: dict[str, int | None] = {"rss_bytes": None, "pss_bytes": None}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                result["rss_bytes"] = int(line.split()[1]) * 1024
                break
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.is_file():
        for line in rollup.read_text(encoding="utf-8").splitlines():
            if line.startswith("Pss:"):
                result["pss_bytes"] = int(line.split()[1]) * 1024
                break
    return result


def release_unused_memory() -> None:
    """釋放無法再觸及的 Python 物件與未使用的 CPU/CUDA allocator pages。"""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _resource_snapshot(trainer: Any, *, event: str, batch_seen: int) -> dict[str, Any]:
    process_memory = _process_memory_bytes()
    payload: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "epoch": int(getattr(trainer, "epoch", -1)) + 1,
        "batch_seen": batch_seen,
        "ram_available_bytes": _memory_available_bytes(),
        "process_rss_bytes": process_memory["rss_bytes"],
        "process_pss_bytes": process_memory["pss_bytes"],
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        payload.update(
            gpu_device=device,
            gpu_name=torch.cuda.get_device_name(device),
            vram_free_bytes=int(free),
            vram_total_bytes=int(total),
            vram_allocated_bytes=int(torch.cuda.memory_allocated(device)),
            vram_reserved_bytes=int(torch.cuda.memory_reserved(device)),
            vram_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        )
    return payload


def _assert_resource_floor(snapshot: dict[str, Any]) -> None:
    available_ram = int(snapshot["ram_available_bytes"])
    if available_ram < RUNTIME_MIN_AVAILABLE_RAM_BYTES:
        raise RuntimeError(
            "可用系統 RAM 低於訓練安全下限："
            f"{available_ram / (1 << 30):.2f} GiB < "
            f"{RUNTIME_MIN_AVAILABLE_RAM_BYTES / (1 << 30):.2f} GiB"
        )
    if snapshot.get("cuda_available"):
        free_vram = int(snapshot["vram_free_bytes"])
        if free_vram < RUNTIME_MIN_FREE_VRAM_BYTES:
            raise RuntimeError(
                "可用 VRAM 低於訓練安全下限："
                f"{free_vram / (1 << 30):.2f} GiB < "
                f"{RUNTIME_MIN_FREE_VRAM_BYTES / (1 << 30):.2f} GiB"
            )


def make_bittrue_validation_copy(model: Any, bittrue_config: Path):
    """Deep-copy Float training state and reconfigure only its attention backend."""

    validation_model = copy.deepcopy(unwrap_model(model))
    convert_yolo26_model(validation_model, VariantConfig.from_yaml(bittrue_config))
    return validation_model


def _assert_finite(model: Any, label: str) -> None:
    bad = [
        name
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
    ]
    if bad:
        raise FloatingPointError(f"non-finite {label} tensors: {bad[:10]}")


class MASFTrainer(DetectionTrainer):
    """Rebuild optimizer/scheduler/EMA per phase and enforce immutable frozen state."""

    def __init__(
        self,
        *args: Any,
        masf_variant: str,
        attention_config: Path,
        bittrue_config: Path,
        phase_spec: PhaseSpec,
        validation_batch: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.masf_variant = masf_variant
        self.attention_config = attention_config
        self.bittrue_config = bittrue_config
        self.phase_spec = phase_spec
        if validation_batch is not None and validation_batch < 1:
            raise ValueError("validation_batch 必須是正整數")
        self.validation_batch = validation_batch
        self._frozen_snapshot: dict[str, torch.Tensor] = {}
        self._resource_batch_count = 0
        super().__init__(*args, **kwargs)
        self._resource_telemetry_path = Path(self.save_dir).resolve().parent / "resource-telemetry.jsonl"
        self.add_callback("on_train_start", self._on_train_start)
        self.add_callback("on_train_epoch_start", self._disable_oom_fallback)
        self.add_callback("on_train_epoch_end", self._on_epoch_end)
        self.add_callback("on_train_batch_end", self._on_batch_end)
        self.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        self.add_callback("on_model_save", self._on_model_save)

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        model, _ = build_training_model(
            cfg=cfg,
            nc=self.data["nc"],
            channels=self.data["channels"],
            weights=weights,
            masf_variant=self.masf_variant,
            attention_config=self.attention_config,
            verbose=verbose,
        )
        return model

    def get_dataloader(self, dataset_path: str, batch_size: int, rank: int = 0, mode: str = "train"):
        """Keep batch fixed and prevent simultaneous validation workers."""

        if mode == "val":
            validation_batch = getattr(self, "validation_batch", None) or self.batch_size
            batch_size = min(batch_size, validation_batch)
            configured_workers = self.args.workers
            self.args.workers = 0
            try:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = configured_workers
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def build_optimizer(self, model: Any, *args: Any, **kwargs: Any):
        apply_phase_scope(model, self.phase_spec.name)
        optimizer = build_phase_optimizer(model, self.phase_spec)
        enforce_frozen_modules_eval(model)
        self._frozen_snapshot = snapshot_frozen_state(model)
        return optimizer

    def resume_training(self, ckpt: dict[str, Any] | None) -> None:
        """恢復 optimizer/EMA，並延續中斷前的 early-stopping 歷史。"""

        super().resume_training(ckpt)
        if ckpt is None or not self.resume:
            return
        best_epoch, csv_best_fitness = _historical_best_fitness(self.csv)
        checkpoint_best_fitness = float(self.best_fitness)
        if not math.isclose(
            csv_best_fitness,
            checkpoint_best_fitness,
            rel_tol=0.0,
            abs_tol=5e-5,
        ):
            raise RuntimeError(
                "續訓 checkpoint 的 best_fitness 與 results.csv 不一致："
                f"{checkpoint_best_fitness} != {csv_best_fitness}"
            )
        self.stopper.best_epoch = best_epoch
        self.stopper.best_fitness = checkpoint_best_fitness

    def _model_train(self) -> None:
        """Enter train mode, then restore immutable frozen BN and attention state."""

        super()._model_train()
        enforce_frozen_modules_eval(unwrap_model(self.model))

    @staticmethod
    def _disable_oom_fallback(trainer: Any) -> None:
        """Prevent upstream first-epoch OOM recovery from changing the fixed batch."""

        trainer._oom_retries = 3

    def _on_epoch_end(self, trainer: Any) -> None:
        assert_frozen_state_unchanged(unwrap_model(trainer.model), self._frozen_snapshot)

    def _record_resources(self, trainer: Any, event: str) -> dict[str, Any]:
        snapshot = _resource_snapshot(trainer, event=event, batch_seen=self._resource_batch_count)
        self._resource_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._resource_telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        return snapshot

    def _on_train_start(self, trainer: Any) -> None:
        snapshot = self._record_resources(trainer, "train_start")
        _assert_resource_floor(snapshot)

    def _on_batch_end(self, trainer: Any) -> None:
        loss = getattr(trainer, "loss", None)
        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite training loss")
        self._resource_batch_count += 1
        if self._resource_batch_count % RESOURCE_CHECK_INTERVAL_BATCHES == 0:
            snapshot = self._record_resources(trainer, "batch_interval")
            _assert_resource_floor(snapshot)

    def _on_fit_epoch_end(self, trainer: Any) -> None:
        self._record_resources(trainer, "epoch_end_before_cleanup")
        release_unused_memory()
        snapshot = self._record_resources(trainer, "epoch_end_after_cleanup")
        _assert_resource_floor(snapshot)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _on_model_save(trainer: Any) -> None:
        _assert_finite(unwrap_model(trainer.model), "live checkpoint")
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema is not None:
            _assert_finite(ema, "EMA checkpoint")

    def validate(self):
        """Select best.pt and drive early stopping with actual Bit-True mAP50-95(B)."""

        float_ema = self.ema.ema
        self.ema.ema = make_bittrue_validation_copy(float_ema or self.model, self.bittrue_config)
        try:
            metrics = self.validator(self)
        finally:
            self.ema.ema = float_ema
        if metrics is None:
            return None, None
        metrics.pop("fitness", None)
        if MAP_SELECTION_METRIC not in metrics:
            raise KeyError(f"validator omitted {MAP_SELECTION_METRIC}")
        fitness = float(metrics[MAP_SELECTION_METRIC])
        if not torch.isfinite(torch.tensor(fitness)):
            raise FloatingPointError("non-finite validation mAP50-95")
        if not self.best_fitness or fitness > self.best_fitness:
            self.best_fitness = fitness
        return metrics, fitness


@dataclass(frozen=True)
class TrainerFactory:
    masf_variant: str
    attention_config: Path
    bittrue_config: Path
    phase_spec: PhaseSpec
    validation_batch: int | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> MASFTrainer:
        return MASFTrainer(
            *args,
            masf_variant=self.masf_variant,
            attention_config=self.attention_config,
            bittrue_config=self.bittrue_config,
            phase_spec=self.phase_spec,
            validation_batch=self.validation_batch,
            **kwargs,
        )


def _historical_best_fitness(path: Path) -> tuple[int, float]:
    """從既有 results.csv 還原 1-based 最佳 epoch 與 Bit-True fitness。"""

    if not path.is_file():
        raise FileNotFoundError(f"續訓缺少歷史 metrics：{path}")
    candidates: list[tuple[int, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(float(row["epoch"]))
            fitness = float(row[MAP_SELECTION_METRIC])
            if epoch < 1 or not math.isfinite(fitness):
                raise ValueError(f"無效的續訓歷史 metrics：{path}")
            candidates.append((epoch, fitness))
    if not candidates:
        raise ValueError(f"續訓歷史 metrics 為空：{path}")
    return max(candidates, key=lambda item: item[1])


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_training_manifest(
    destination: Path,
    *,
    project_root: Path,
    weights: Path,
    variant: str,
    phase: PhaseSpec,
    common: CommonTrainingConfig,
    validation_batch: int | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "phase": asdict(phase),
        "common": asdict(common),
        "parent": {"path": str(weights.resolve()), "sha256": file_sha256(weights)},
        "fresh_optimizer_scheduler_ema": True,
        "automatic_batch_fallback": False,
        "gradient_accumulation": common.gradient_accumulation,
        "in_training_validation_batch": validation_batch or common.batch,
        "resource_safety": {
            "check_interval_batches": RESOURCE_CHECK_INTERVAL_BATCHES,
            "minimum_available_ram_bytes": RUNTIME_MIN_AVAILABLE_RAM_BYTES,
            "minimum_free_vram_bytes": RUNTIME_MIN_FREE_VRAM_BYTES,
            "epoch_end_gc": True,
            "epoch_end_cuda_empty_cache": True,
            "epoch_end_malloc_trim": True,
            "telemetry": "resource-telemetry.jsonl",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": nvidia_driver_version(),
            "gpu_vram_bytes": (
                torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
            ),
            "gpu_free_vram_bytes_at_start": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
            "ultralytics": ultralytics.__version__,
            "ultralytics_source": ultralytics.__path__[0],
            "git_revision": _git_revision(project_root),
        },
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _write_training_complete(
    run: Path,
    *,
    phase: PhaseSpec,
    trainer: Any,
    elapsed_seconds: float,
    resumed_from: Path | None = None,
    resumed_from_sha256: str | None = None,
    resumed_after_epoch: int | None = None,
) -> Path:
    """以原子替換寫入 phase 完成證據。"""

    completed_epochs = int(getattr(trainer, "epoch", -1)) + 1
    payload: dict[str, Any] = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "requested_epochs": phase.epochs,
        "completed_epochs": completed_epochs,
        "patience": phase.patience,
        "stop_reason": "early-stopping" if completed_epochs < phase.epochs else "max-epochs",
        "elapsed_seconds": elapsed_seconds,
        "best_fitness": float(trainer.best_fitness),
        "best_checkpoint": str((run / "ultralytics/weights/best.pt").resolve()),
        "epoch_metrics_csv": str((run / "ultralytics/results.csv").resolve()),
    }
    if resumed_from is not None:
        payload["resumed_from"] = {
            "checkpoint": str(resumed_from.resolve()),
            "sha256": resumed_from_sha256,
            "completed_epoch": resumed_after_epoch,
        }
    destination = run / "training-complete.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def launch_phase(
    *,
    project_root: Path,
    weights: Path,
    masf_variant: str,
    attention_config: Path,
    bittrue_config: Path,
    phase: PhaseSpec,
    common: CommonTrainingConfig,
    run_id: str,
    validation_batch: int | None = None,
) -> Any:
    """Launch exactly one phase from its accepted parent checkpoint."""

    run = project_root / "artifacts" / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    write_training_manifest(
        run / "manifest.json",
        project_root=project_root,
        weights=weights,
        variant=masf_variant,
        phase=phase,
        common=common,
        validation_batch=validation_batch,
    )
    args = common.to_ultralytics_args(phase, project=run, name="ultralytics")
    data = Path(args["data"])
    if not data.is_absolute():
        args["data"] = str((project_root / data).resolve())
    model = YOLO(str(weights.resolve()))
    started = time.perf_counter()
    try:
        result = model.train(
            trainer=TrainerFactory(
                masf_variant,
                attention_config.resolve(),
                bittrue_config.resolve(),
                phase,
                validation_batch,
            ),
            **args,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"formal batch={common.batch} does not fit; stopped without automatic batch change"
        ) from exc
    finally:
        release_unused_memory()
    elapsed = time.perf_counter() - started
    _write_training_complete(
        run,
        phase=phase,
        trainer=model.trainer,
        elapsed_seconds=elapsed,
    )
    return result


def resume_phase(
    *,
    project_root: Path,
    masf_variant: str,
    attention_config: Path,
    bittrue_config: Path,
    phase: PhaseSpec,
    common: CommonTrainingConfig,
    run_id: str,
    validation_batch: int | None = None,
) -> Any:
    """從既有 run 的 last.pt 恢復因主機中斷而未完成的 phase。"""

    run = (project_root / "artifacts" / "runs" / run_id).resolve()
    manifest_path = run / "manifest.json"
    complete_path = run / "training-complete.json"
    last = run / "ultralytics/weights/last.pt"
    if complete_path.exists():
        raise FileExistsError(f"phase 已完成，不可續訓：{complete_path}")
    if not manifest_path.is_file() or not last.is_file():
        raise FileNotFoundError(f"中斷 run 缺少 manifest.json 或 last.pt：{run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("variant") != masf_variant
        or manifest.get("phase") != asdict(phase)
        or manifest.get("common") != asdict(common)
        or manifest.get("in_training_validation_batch", common.batch)
        != (validation_batch or common.batch)
    ):
        raise RuntimeError(f"中斷 run 與目前實驗契約不一致：{run}")

    model = YOLO(str(last))
    checkpoint = model.ckpt or {}
    resumed_after_epoch = int(checkpoint.get("epoch", -1)) + 1
    if (
        resumed_after_epoch < 1
        or resumed_after_epoch >= phase.epochs
        or checkpoint.get("optimizer") is None
        or checkpoint.get("ema") is None
    ):
        raise RuntimeError(f"last.pt 不含有效的續訓狀態：{last}")

    resume_checkpoint_sha256 = file_sha256(last)
    history = run / "resume-history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": str(last),
                    "checkpoint_sha256": resume_checkpoint_sha256,
                    "completed_epoch": resumed_after_epoch,
                    "reason": "interrupted-run-recovery",
                },
                sort_keys=True,
            )
            + "\n"
        )

    args = common.to_ultralytics_args(phase, project=run, name="ultralytics")
    args["resume"] = str(last)
    started = time.perf_counter()
    try:
        result = model.train(
            trainer=TrainerFactory(
                masf_variant,
                attention_config.resolve(),
                bittrue_config.resolve(),
                phase,
                validation_batch,
            ),
            **args,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"formal batch={common.batch} does not fit; stopped without automatic batch change"
        ) from exc
    finally:
        release_unused_memory()
    elapsed = time.perf_counter() - started
    _write_training_complete(
        run,
        phase=phase,
        trainer=model.trainer,
        elapsed_seconds=elapsed,
        resumed_from=last,
        resumed_from_sha256=resume_checkpoint_sha256,
        resumed_after_epoch=resumed_after_epoch,
    )
    return result
