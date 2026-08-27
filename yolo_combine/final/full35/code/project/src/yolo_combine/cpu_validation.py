"""CPU-only acceptance checks for one isolated architecture workspace."""

from __future__ import annotations

import copy
import gc
from dataclasses import asdict
from typing import Any

import torch
import ultralytics

from .data import audit_bbt5
from .freezing import InheritedFreezeGuard
from .materialize import build_validation_models
from .models import RoutedDualModel, SharedDualHeadModel
from .source import SourceBundle
from .training import TaskLossRouter
from .variants import VariantWorkspace


def _assert_nested_exact(actual: Any, expected: Any) -> tuple[int, int]:
    """Assert bit-exact nested outputs and return tensor/element counts."""

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return 1, expected.numel()
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(
                f"mapping keys differ: {actual.keys()} != {expected.keys()}"
            )
        counts = [_assert_nested_exact(actual[key], expected[key]) for key in expected]
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError("nested output sequence structure differs")
        counts = [
            _assert_nested_exact(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        ]
    else:
        if actual != expected:
            raise AssertionError(f"nested scalar differs: {actual!r} != {expected!r}")
        return 0, 0
    return sum(value[0] for value in counts), sum(value[1] for value in counts)


def _synthetic_detect_batch(imgsz: int) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, imgsz, imgsz),
        "batch_idx": torch.zeros(1, dtype=torch.long),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
    }


def _synthetic_pose_batch(imgsz: int) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, imgsz, imgsz),
        "batch_idx": torch.zeros(1, dtype=torch.long),
        "cls": torch.ones(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.2]]),
        "keypoints": torch.tensor([[[0.4, 0.5, 2.0], [0.6, 0.5, 2.0]]]),
    }


def _validate_bittrue_inference(
    source: SourceBundle,
    *,
    imgsz: int,
    seed: int,
) -> dict[str, Any]:
    """Build the retained Bit-True graph and prove shared forward equivalence on CPU."""

    torch.manual_seed(seed)
    built = source.build_task_models(kind="bittrue")
    independent_parameters = sum(
        parameter.numel()
        for model in (built.detect, built.pose)
        for parameter in model.parameters()
    )
    detect = copy.deepcopy(built.detect).cpu().eval()
    pose = copy.deepcopy(built.pose).cpu().eval()
    routed = RoutedDualModel(detect, pose)
    shared = SharedDualHeadModel.from_task_models(built.detect, built.pose).cpu().eval()
    input_tensor = torch.rand(1, 3, imgsz, imgsz)
    with torch.inference_mode():
        expected = routed(input_tensor)
        actual = shared(input_tensor)
    exact = {
        task: dict(
            zip(
                ("tensor_leaves", "elements"),
                _assert_nested_exact(actual[task], expected[task]),
                strict=True,
            )
        )
        for task in ("detect", "pose")
    }
    shared_parameters = sum(parameter.numel() for parameter in shared.parameters())
    return {
        "contract": shared.contract(),
        "trunk_transfer": asdict(built.transfer),
        "independent_parameters": independent_parameters,
        "shared_parameters": shared_parameters,
        "removed_duplicate_parameters": independent_parameters - shared_parameters,
        "exact_vs_independent": exact,
    }


def validate_workspace_on_cpu(
    workspace: VariantWorkspace,
    *,
    imgsz: int = 64,
    seed: int = 0,
    joint_step: bool = True,
) -> dict[str, Any]:
    """Exercise source, data, models, exact forwards, export seam, and backward on CPU."""

    if imgsz < 64 or imgsz % 32:
        raise ValueError("imgsz must be a multiple of 32 and at least 64")
    audit = workspace.audit(verify_hashes=True)
    if not audit.ok:
        raise RuntimeError(f"workspace audit failed: {audit}")

    torch.manual_seed(seed)
    source = SourceBundle(workspace.source_bundle, architecture=workspace.architecture)
    manifest = source.verify_manifest()
    dataset = audit_bbt5(workspace.bbat5_registry)
    if not dataset.derivation_exact or dataset.broken_image_links:
        raise RuntimeError(f"BBT5 source views failed validation: {dataset}")

    built = source.build_task_models()
    independent_parameters = sum(
        parameter.numel()
        for model in (built.detect, built.pose)
        for parameter in model.parameters()
    )
    independent_detect = copy.deepcopy(built.detect).cpu().eval()
    independent_pose = copy.deepcopy(built.pose).cpu().eval()
    routed = RoutedDualModel(independent_detect, independent_pose)
    shared = SharedDualHeadModel.from_task_models(built.detect, built.pose).cpu().eval()
    shared_parameters = sum(parameter.numel() for parameter in shared.parameters())
    input_tensor = torch.rand(1, 3, imgsz, imgsz)

    with torch.inference_mode():
        routed_output = routed(input_tensor)
        shared_output = shared(input_tensor)
    independent_exact = {
        task: dict(
            zip(
                ("tensor_leaves", "elements"),
                _assert_nested_exact(shared_output[task], routed_output[task]),
                strict=True,
            )
        )
        for task in ("detect", "pose")
    }
    del routed_output, independent_detect, independent_pose, routed
    gc.collect()

    materialized = build_validation_models(shared, source)
    materialized.detect.cpu().eval()
    materialized.pose.cpu().eval()
    with torch.inference_mode():
        detect_output = materialized.detect(input_tensor)
        pose_output = materialized.pose(input_tensor)
    materialized_exact = {
        "detect": dict(
            zip(
                ("tensor_leaves", "elements"),
                _assert_nested_exact(detect_output, shared_output["detect"]),
                strict=True,
            )
        ),
        "pose": dict(
            zip(
                ("tensor_leaves", "elements"),
                _assert_nested_exact(pose_output, shared_output["pose"]),
                strict=True,
            )
        ),
    }
    materialization = {
        "detect_complete": materialized.detect_report.complete,
        "detect_tensors": materialized.detect_report.compatible_tensors,
        "pose_complete": materialized.pose_report.complete,
        "pose_tensors": materialized.pose_report.compatible_tensors,
        "exact_outputs": materialized_exact,
    }
    del detect_output, pose_output, materialized, shared_output
    gc.collect()

    joint: dict[str, Any] | None = None
    if joint_step:
        shared.train()
        guard = InheritedFreezeGuard.capture(shared)
        router = TaskLossRouter(shared, epochs=1, imgsz=imgsz)
        trainable = [
            parameter for parameter in shared.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(trainable, lr=1e-5)
        step = router.optimizer_step(
            optimizer,
            detect_batches=[
                _synthetic_detect_batch(imgsz),
                _synthetic_detect_batch(imgsz),
            ],
            pose_batches=[_synthetic_pose_batch(imgsz)],
        )
        guard.assert_unchanged(shared)
        finite_gradient_tensors = sum(
            1
            for parameter in trainable
            if parameter.grad is not None and torch.isfinite(parameter.grad).all()
        )
        joint = {
            **asdict(step),
            "ratio": "2:1",
            "finite_gradient_tensors": finite_gradient_tensors,
            "inherited_paths": list(guard.paths),
        }

    reduction = independent_parameters - shared_parameters
    bittrue = _validate_bittrue_inference(source, imgsz=imgsz, seed=seed)
    return {
        "schema_version": 1,
        "architecture": workspace.architecture,
        "device": "cpu",
        "seed": seed,
        "imgsz": imgsz,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "workspace": {
            "root": str(workspace.root),
            "run_root": str(workspace.run_root),
            "audit": asdict(audit),
        },
        "source_manifest": asdict(manifest),
        "bbt5": {
            "train": asdict(dataset.train),
            "valid": asdict(dataset.valid),
            "derivation_mismatches": dataset.derivation_mismatches,
            "broken_image_links": dataset.broken_image_links,
            "source_group_overlap": list(dataset.source_group_overlap),
        },
        "model": {
            "contract": shared.contract(),
            "trunk_transfer": asdict(built.transfer),
            "independent_parameters": independent_parameters,
            "shared_parameters": shared_parameters,
            "removed_duplicate_parameters": reduction,
            "parameter_reduction_fraction": reduction / independent_parameters,
            "exact_vs_independent": independent_exact,
            "materialization": materialization,
            "bittrue_inference": bittrue,
        },
        "joint_optimizer_step": joint,
    }
