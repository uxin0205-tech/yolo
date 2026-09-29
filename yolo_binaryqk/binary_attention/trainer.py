"""Ultralytics trainer integration with fail-closed binary and KD diagnostics."""
from __future__ import annotations

from pathlib import Path
import hashlib
import math
import os
from types import MethodType

import numpy as np
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.block import Attention
from ultralytics.data.build import (
    ContiguousDistributedSampler,
    seed_worker,
)
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from torch.utils.data import DataLoader, distributed


FORMAL_BATCH = 128
DEFAULT_MICRO_BATCH = 16
MICRO_BATCH_ENV = "BINARYQK_MICRO_BATCH"


def _fp_attention_probability(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Reconstruct the frozen Ultralytics FP attention map from module input."""

    batch, _channels, height, width = x.shape
    tokens = height * width
    qkv = module.qkv(x).view(batch, module.num_heads, module.key_dim * 2 + module.head_dim, tokens)
    q, k, _v = qkv.split([module.key_dim, module.key_dim, module.head_dim], dim=2)
    return ((q * module.scale).transpose(-2, -1) @ k).softmax(dim=-1)


def freeze_to_attention_only(model: torch.nn.Module) -> tuple[str, ...]:
    """Freeze every parameter except concrete attention-module parameters."""

    model.requires_grad_(False)
    trainable: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, Attention):
            continue
        for parameter_name, parameter in module.named_parameters(recurse=True):
            parameter.requires_grad_(True)
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            trainable.append(full_name)
    names = tuple(sorted(set(trainable)))
    if not names:
        raise RuntimeError("attention-only fine-tuning found no concrete Attention parameters")
    return names


def _frozen_teacher_prediction(teacher: torch.nn.Module, images: torch.Tensor):
    """Return raw Detect logits without changing frozen child-module modes."""

    teacher_head = teacher.model[-1]
    teacher_head_training = teacher_head.training
    teacher_head.training = True
    try:
        with torch.no_grad():
            return teacher(images)
    finally:
        teacher_head.training = teacher_head_training


def _validation_counter_model(trainer) -> torch.nn.Module:
    """Return the exact model Ultralytics passes to its training validator."""

    ema = getattr(getattr(trainer, "ema", None), "ema", None)
    return ema if ema is not None else trainer.model


def resolve_runtime_batch(requested_batch: int) -> tuple[int, int]:
    """Return ``(micro_batch, effective_batch)`` for the formal run.

    The formal plan compares an effective batch of 128.  This host's 16-GiB
    RTX 4080 SUPER cannot hold YOLO11m at 640px with 128 images in one CUDA
    forward, so the optimizer sees eight 16-image micro-batches instead.  The
    limit is overridable for a host with more VRAM while the recorded formal
    batch remains unchanged.
    """

    if requested_batch < 1:
        return requested_batch, requested_batch
    try:
        limit = int(os.environ.get(MICRO_BATCH_ENV, DEFAULT_MICRO_BATCH))
    except ValueError as exc:
        raise ValueError(f"{MICRO_BATCH_ENV} must be a positive integer") from exc
    if limit < 1:
        raise ValueError(f"{MICRO_BATCH_ENV} must be a positive integer")
    if requested_batch <= limit:
        return requested_batch, requested_batch
    return limit, requested_batch


def _identity_collate(items):
    """Keep one decoded sample per worker; the training wrapper collates batches."""

    return items[0]


class _BatchAccumulatingLoader:
    """Stream single samples from workers and collate the requested batch in the parent.

    PyTorch workers otherwise collate a full batch in every worker.  For the
    required COCO batch=128 this multiplies decoded-image memory by the
    worker count.  Each worker still participates (``num_workers=8``), while
    the effective model batch remains 128 and only one sample is prefetched
    per worker.
    """

    def __init__(self, loader: DataLoader, dataset, batch_size: int):
        self._loader = loader
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = loader.num_workers
        self.sampler = loader.sampler
        self._sample_count = len(loader)

    def __len__(self):
        return math.ceil(self._sample_count / self.batch_size)

    def __iter__(self):
        samples = []
        for sample in self._loader:
            samples.append(sample)
            if len(samples) == self.batch_size:
                yield self.dataset.collate_fn(samples)
                samples = []
        if samples:
            yield self.dataset.collate_fn(samples)

    def reset(self):
        """Match Ultralytics' InfiniteDataLoader API; the next iter rebuilds workers."""

        return None

    def close(self):
        """Shut down persistent worker processes when Ultralytics tears down."""

        iterator = getattr(self._loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            self._loader._iterator = None

from .kd import add_kd_loss
from .model import build_student
from .variants.definitions import VariantDefinition, get_variant
from .verification.checkpoint import binary_modules


class BinaryDetectionTrainer(DetectionTrainer):
    """Train a parser-built variant with optional frozen FP teacher."""

    def __init__(
        self,
        *args,
        variant: VariantDefinition,
        model_yaml: Path,
        source_weights: Path | None = None,
        init_weights: Path | None = None,
        max_batches: int | None = None,
        **kwargs,
    ):
        self.variant = variant
        self.model_yaml = Path(model_yaml)
        self.source_weights = Path(source_weights) if source_weights else None
        self.init_weights = Path(init_weights) if init_weights else None
        self.teacher: torch.nn.Module | None = None
        self.teacher_hash: str | None = None
        self.kd_weight = 0.1
        self.last_kd_loss = torch.tensor(0.0)
        self._counter_at_batch_start = 0
        self._expected_binary_paths: tuple[str, ...] | None = None
        self._trainable_parameter_names: tuple[str, ...] = ()
        self.max_batches = max_batches
        self._seen_batches = 0
        super().__init__(*args, **kwargs)
        self.requested_batch = int(self.args.batch)
        self.micro_batch, self.effective_batch = resolve_runtime_batch(self.requested_batch)
        if self.requested_batch > 0 and self.micro_batch < self.requested_batch:
            self.args.batch = self.batch_size = self.micro_batch
            # Make the optimizer and warmup schedule accumulate back to the
            # formal batch instead of silently changing the experiment to 64.
            self.args.nbs = max(int(self.args.nbs), self.effective_batch)
        if max_batches is not None:
            self.add_callback("on_train_batch_end", self._stop_after_batches)
        self.add_callback("on_train_epoch_end", self._check_epoch_binary_structure)

    def execution_profile(self) -> dict[str, int]:
        """Return the runtime batch details persisted beside formal args."""

        accumulation = max(round(self.effective_batch / self.batch_size), 1) if self.batch_size > 0 else 1
        return {
            "requested_batch": self.requested_batch,
            "micro_batch": self.batch_size,
            "effective_batch": self.effective_batch,
            "gradient_accumulation_steps": accumulation,
            "trainable_scope": "attention_only",
        }

    def trainable_profile(self) -> dict[str, int | str | list[str]]:
        """Return auditable attention-only parameter counts after optimizer setup."""

        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        attention_paths = tuple(name for name, module in self.model.named_modules() if isinstance(module, Attention))
        frozen_batchnorm_count = sum(
            1
            for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.BatchNorm2d)
            and not any(name == path or name.startswith(f"{path}.") for path in attention_paths)
        )
        return {
            "trainable_scope": "attention_only",
            "trainable_parameter_count": trainable,
            "frozen_parameter_count": total - trainable,
            "total_parameter_count": total,
            "trainable_parameter_names": list(self._trainable_parameter_names),
            "frozen_non_attention_batchnorm_count": frozen_batchnorm_count,
        }

    def build_optimizer(self, model, *args, **kwargs):
        """Apply the inverse freeze immediately before optimizer construction."""

        self._trainable_parameter_names = freeze_to_attention_only(model)
        return super().build_optimizer(model, *args, **kwargs)

    def _model_train(self):
        """Train attention while keeping frozen backbone/head BN statistics fixed."""

        super()._model_train()
        attention_paths = tuple(name for name, module in self.model.named_modules() if isinstance(module, Attention))
        for name, module in self.model.named_modules():
            inside_attention = any(name == path or name.startswith(f"{path}.") for path in attention_paths)
            if isinstance(module, torch.nn.BatchNorm2d) and not inside_attention:
                module.eval()

    def _stop_after_batches(self, trainer) -> None:
        self._seen_batches += 1
        if self._seen_batches >= self.max_batches:
            self.stop = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Resume rebuilds the same concrete architecture before restoring the
        # native checkpoint state.  This preserves counters and new parameters.
        if self.resume and weights is not None:
            model = build_student(self.model_yaml, self.variant, nc=self.data.get("nc", 80))
            model.load_state_dict(weights.float().state_dict(), strict=True)
            return model
        return build_student(
            self.model_yaml,
            self.variant,
            self.init_weights or self.source_weights,
            nc=self.data.get("nc", 80),
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Use the baseline loader with bounded host-memory prefetching.

        Ultralytics 8.4.90 hard-codes ``prefetch_factor=4``.  With the formal
        plan's required batch=128 and workers=8, that can retain several
        thousand decoded COCO images and exceed the host cgroup limit before
        the first epoch.  Prefetching is an execution-memory setting, not a
        training hyperparameter; keeping it at one preserves the requested
        batch, workers, cache, augmentation and optimizer configuration for
        every variant while making the full-COCO run reproducible on this host.
        """

        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            from ultralytics.utils import LOGGER

            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False

        dataset_len = len(dataset)
        batch_limit = self.micro_batch if self.micro_batch > 0 else batch_size
        batch = min(batch_size, batch_limit, dataset_len)
        sampler = (
            None
            if rank == -1
            else distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else ContiguousDistributedSampler(dataset)
        )
        samples = len(sampler) if sampler is not None else dataset_len
        drop_last = self.args.compile and mode == "train" and bool(batch) and dataset_len % batch != 0
        batches = (samples // batch if drop_last else math.ceil(samples / batch)) if batch else 0
        device_count = torch.cuda.device_count()
        # Validation is already serialized by the trainer and its loader is
        # retained for every epoch.  Spawning the usual ``workers * 2`` here
        # would duplicate the dataset beside the persistent training workers
        # and create a large host-RAM spike at the first validation pass.
        workers = self.args.workers if mode == "train" else 0
        num_workers = min(os.cpu_count() // max(device_count, 1), workers, 0 if batches <= 1 else batches)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + rank)
        sample_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=shuffle and sampler is None,
            num_workers=num_workers,
            sampler=sampler,
            prefetch_factor=1 if num_workers > 0 else None,
            pin_memory=device_count > 0,
            collate_fn=_identity_collate,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
        return _BatchAccumulatingLoader(sample_loader, dataset, batch)

    def setup_model(self):
        result = super().setup_model()
        modules = binary_modules(self.model)
        if self.variant.qk_mode != "fp" and not modules:
            raise RuntimeError("BinaryDetectionTrainer refused FP-only model: no binary attention modules")
        if self.variant.qk_mode == "fp" and modules:
            raise RuntimeError("FP control unexpectedly contains binary attention modules")
        self._expected_binary_paths = tuple(name for name, _ in modules)
        return result

    def _check_binary_structure(self, model) -> None:
        modules = binary_modules(model)
        paths = tuple(name for name, _ in modules)
        if self.variant.qk_mode == "fp" and modules:
            raise RuntimeError("FP control unexpectedly contains binary attention modules")
        if self.variant.qk_mode != "fp" and paths != self._expected_binary_paths:
            raise RuntimeError(f"binary module paths changed: expected {self._expected_binary_paths}, got {paths}")

    def _check_epoch_binary_structure(self, trainer) -> None:
        self._check_binary_structure(self.model)
        if self.variant.qk_mode != "fp":
            count = sum(int(module.binary_forward_count.item()) for _, module in binary_modules(self.model))
            if count <= 0:
                raise RuntimeError("binary_forward_count did not increase during the epoch")

    def preprocess_batch(self, batch):
        self._counter_at_batch_start = sum(int(module.binary_forward_count.item()) for _, module in binary_modules(self.model))
        return super().preprocess_batch(batch)

    def label_loss_items(self, loss_items=None, prefix="train"):
        if loss_items is not None and self.variant.qk_mode != "fp":
            current = sum(int(module.binary_forward_count.item()) for _, module in binary_modules(self.model))
            if current <= self._counter_at_batch_start:
                raise RuntimeError("binary_forward_count did not advance during the training batch")
        return super().label_loss_items(loss_items, prefix)

    def attach_teacher(self, teacher: torch.nn.Module, checkpoint: Path | None = None) -> None:
        """Attach a frozen teacher; it is never included in optimizer or EMA."""

        teacher.eval()
        teacher.requires_grad_(False)
        self.teacher = teacher
        self.teacher_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None

    def _install_kd_loss(self) -> None:
        if not self.variant.use_distillation:
            return
        teacher_weights = self.source_weights
        if not teacher_weights:
            raise ValueError("KD variants require the frozen FP source checkpoint")
        teacher = build_student(self.model_yaml, get_variant("T0"), teacher_weights, nc=self.data.get("nc", 80))
        teacher = teacher.to(self.device)
        self.attach_teacher(teacher, teacher_weights)
        original_loss = self.model.loss
        model = self.model
        student_features: list[torch.Tensor] = []
        teacher_features: list[torch.Tensor] = []
        student_attention: list[torch.Tensor] = []
        teacher_attention: list[torch.Tensor] = []
        student_positional: list[torch.Tensor] = []
        teacher_positional: list[torch.Tensor] = []
        teacher_modules = dict(teacher.named_modules())
        for name, student_module in binary_modules(model):
            teacher_module = teacher_modules.get(name)
            if teacher_module is None:
                continue
            if "feature" in self.variant.kd_components:
                student_module.register_forward_hook(lambda _m, _args, output: student_features.append(output))
                teacher_module.register_forward_hook(lambda _m, _args, output: teacher_features.append(output))
            if "attention" in self.variant.kd_components:
                def capture_student_attention(module, _args, _output):
                    probability = getattr(module, "kd_probabilities", None)
                    if isinstance(probability, torch.Tensor):
                        student_attention.append(probability)
                    module.kd_probabilities = None

                student_module.register_forward_hook(capture_student_attention)
                teacher_module.register_forward_pre_hook(
                    lambda module, args: teacher_attention.append(_fp_attention_probability(module, args[0]))
                )
            if "positional" in self.variant.kd_components:
                student_module.pe.register_forward_hook(
                    lambda _m, _args, output: student_positional.append(output)
                )
                teacher_module.pe.register_forward_hook(
                    lambda _m, _args, output: teacher_positional.append(output)
                )

        def loss_with_kd(_model, batch, preds=None):
            student_features.clear()
            teacher_features.clear()
            student_attention.clear()
            teacher_attention.clear()
            student_positional.clear()
            teacher_positional.clear()
            if preds is None:
                preds = model.forward(batch["img"])
            detection_loss, loss_items = original_loss(batch, preds)
            # Keep the FP teacher's backbone/normalization layers in eval mode,
            # but make only the Detect head emit the same raw training
            # prediction dictionary as the student.  Calling the full teacher
            # in eval mode returns decoded inference tuples, which cannot be
            # paired with the student's training outputs for KD.
            teacher_prediction = _frozen_teacher_prediction(self.teacher, batch["img"])
            feature_pairs = [
                (student, teacher_feature)
                for student, teacher_feature in zip(student_features, teacher_features)
                if isinstance(student, torch.Tensor)
                and isinstance(teacher_feature, torch.Tensor)
                and student.shape == teacher_feature.shape
            ]
            attention_pairs = [
                (student, teacher_map)
                for student, teacher_map in zip(student_attention, teacher_attention)
                if student.shape == teacher_map.shape
            ]
            positional_pairs = [
                (student, teacher_output)
                for student, teacher_output in zip(student_positional, teacher_positional)
                if isinstance(student, torch.Tensor)
                and isinstance(teacher_output, torch.Tensor)
                and student.shape == teacher_output.shape
            ]
            total, kd_value = add_kd_loss(
                detection_loss.sum(), preds, teacher_prediction,
                self.variant.kd_components, self.kd_weight, feature_pairs, attention_pairs,
                positional_pairs,
            )
            self.last_kd_loss = kd_value
            return total, loss_items

        self.model.loss = MethodType(loss_with_kd, self.model)

    def _setup_train(self):
        super()._setup_train()
        self._install_kd_loss()

    def validate(self):
        validation_model = _validation_counter_model(self)
        before = sum(int(module.binary_forward_count.item()) for _, module in binary_modules(validation_model))
        # DetectionValidator constructs its own loader and uses the formal
        # worker count directly.  Keep the recorded training profile at
        # workers=8, but use the parent-process validator path to avoid eight
        # full validation batches being resident at once on this 15-GiB host.
        validator_workers = getattr(getattr(self, "validator", None), "args", None)
        old_workers = getattr(validator_workers, "workers", None)
        if validator_workers is not None:
            validator_workers.workers = 0
        try:
            result = super().validate()
        finally:
            if validator_workers is not None and old_workers is not None:
                validator_workers.workers = old_workers
        after = sum(int(module.binary_forward_count.item()) for _, module in binary_modules(validation_model))
        if self.variant.qk_mode != "fp" and after <= before:
            raise RuntimeError("validation did not execute the binary attention path")
        return result
