from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from activation_lab import build_activation
from activation_lab.hardware import hardware_proxy
from activation_lab.training import StaticPolicy, load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/full35_nonsilu_q3.py"
MANIFEST_PATH = PROJECT_ROOT / "training/full35/contracts/activation-manifest.yaml"


def _load_runner() -> ModuleType:
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("full35_nonsilu_q3", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dyadic_leaky_relu_is_parameter_free_and_shift_friendly() -> None:
    module = build_activation("leaky_relu_1_8")
    values = module(torch.tensor([-8.0, -1.0, 0.0, 2.0]))

    assert torch.equal(values, torch.tensor([-1.0, -0.125, 0.0, 2.0]))
    assert list(module.parameters()) == []
    proxy = hardware_proxy("leaky_relu_1_8")
    assert proxy.variable_multiplications == 0
    assert proxy.constant_multiplications == 0
    assert proxy.dyadic_or_apot_constants
    assert proxy.range_operations == 1


def test_q3_region_policy_covers_all_190_sites_without_silu() -> None:
    runner = _load_runner()
    manifest = load_manifest(MANIFEST_PATH)
    policy = StaticPolicy(
        policy_id="test--all-nonsilu",
        default_activation="qsilu_pq",
        region_assignments=(("neck_attention", "hardswish"),),
    )

    counts = runner._audit_all_nonsilu(manifest, policy)

    assert sum(counts.values()) == 190
    assert counts == {"qsilu_pq": 189, "hardswish": 1}
    assert "silu" not in counts


def test_q3_rejects_silu_relu_and_qsilu_as_simpler_candidates() -> None:
    runner = _load_runner()

    for candidate in ("silu", "relu", "qsilu_pq"):
        with pytest.raises(ValueError, match="cannot contain"):
            runner._validate_candidates((candidate,))


def test_q3_region_metadata_marks_task_specific_regions_not_applicable() -> None:
    runner = _load_runner()

    metadata = runner._region_metadata()

    detect_only = metadata["detect_one2many"]
    assert detect_only["detect_applicable"] is True
    assert detect_only["pose_applicable"] is False
    assert detect_only["pose_deployment_fraction"] == 0.0
    assert detect_only["pose_site_count"] == 0

    pose_only = metadata["pose_flow"]
    assert pose_only["detect_applicable"] is False
    assert pose_only["pose_applicable"] is True
    assert pose_only["detect_deployment_fraction"] == 0.0
    assert pose_only["detect_site_count"] == 0


def test_q3_runner_contains_no_recovery_or_training_call() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert ".run_recovery(" not in source
    assert "FormalJointTrainingSession" not in source
    assert "optimizer" in source  # The generated report explicitly says none was used.
    assert "plots=False" in source


def test_q3_hides_cuda_before_importing_torch() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    guard_index = source.index('os.environ["CUDA_VISIBLE_DEVICES"] = ""')
    torch_import_index = source.index("import torch")

    assert guard_index < torch_import_index
    runner = _load_runner()
    assert runner._assert_cpu_only() == {
        "cuda_visible_devices": "",
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
    }
