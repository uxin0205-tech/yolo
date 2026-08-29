from __future__ import annotations

from pathlib import Path

from activation_lab.training import (
    TrainingArchitecture,
    load_training_config,
    load_yaml_mapping,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_delivery() -> dict:
    digest_a = "a" * 64
    digest_b = "b" * 64
    digest_c = "c" * 64
    return {
        "schema_version": 1,
        "delivery_id": "immutable-delivery-001",
        "model": {
            "source_id": "model-source-001",
            "config_path": "/delivery/model.yaml",
            "checkpoint_path": "/delivery/baseline.pt",
            "checkpoint_sha256": digest_a,
            "framework_commit": "deadbeef",
            "python_version": "3.12.3",
            "torch_version": "2.11.0",
            "framework_version": "confirmed-on-delivery",
        },
        "activation_manifest_path": "/delivery/activation-manifest.yaml",
        "datasets": [
            {
                "dataset_id": "bbat5-v1-detect",
                "yaml_path": "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml",
                "dataset_fingerprint": "bbat5-fingerprint",
                "baseline_checkpoint_path": "/delivery/bbat5.pt",
                "baseline_checkpoint_sha256": digest_b,
                "baseline_metrics": {"map50_95": 0.25},
                "recipe_path": "/delivery/bbat5-recipe.yaml",
            },
            {
                "dataset_id": "coco2017-detect",
                "yaml_path": "/home/uxin/yolo/coco2017.yaml",
                "dataset_fingerprint": "coco-fingerprint",
                "baseline_checkpoint_path": "/delivery/coco.pt",
                "baseline_checkpoint_sha256": digest_c,
                "baseline_metrics": {"map50_95": 0.40, "ap_s": 0.20},
                "recipe_path": "/delivery/coco-recipe.yaml",
            },
        ],
        "hardware_target": None,
    }


def test_delivery_example_cannot_accidentally_pass_as_real_delivery() -> None:
    config = load_training_config(PROJECT_ROOT / "training/configs/pipeline.yaml")
    architecture = TrainingArchitecture(config)
    example = load_yaml_mapping(
        PROJECT_ROOT / "training/contracts/delivery.example.yaml"
    )
    audit = architecture.audit_delivery(example, check_files=False)
    assert not audit.ok
    assert any("placeholder" in error or "numeric" in error for error in audit.errors)


def test_complete_delivery_contract_passes_structure_audit_without_hardware_claim() -> (
    None
):
    config = load_training_config(PROJECT_ROOT / "training/configs/pipeline.yaml")
    audit = TrainingArchitecture(config).audit_delivery(
        _valid_delivery(),
        check_files=False,
    )
    assert audit.ok
    assert audit.errors == ()
    assert audit.warnings == (
        "hardware_target 尚未提供；只能產生結構 proxy，不能宣稱 latency/area/power",
    )


def test_delivery_cannot_swap_coco_and_bbat5_yaml() -> None:
    config = load_training_config(PROJECT_ROOT / "training/configs/pipeline.yaml")
    delivery = _valid_delivery()
    delivery["datasets"][0]["yaml_path"] = "/home/uxin/yolo/coco2017.yaml"
    audit = TrainingArchitecture(config).audit_delivery(delivery, check_files=False)
    assert not audit.ok
    assert any("canonical YAML" in error for error in audit.errors)
