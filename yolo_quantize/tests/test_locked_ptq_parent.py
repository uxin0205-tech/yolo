import hashlib
from types import SimpleNamespace

import pytest

from yolo_quantize.mixed_policy_search import (
    _packed_cost,
    add_inherited_cost,
    build_locked_parent,
)
from yolo_quantize.progressive_preparation import LockedQATParentSpec
from yolo_quantize.qat_runtime import Full35QATRuntime


def setup_parent(tmp_path, monkeypatch):
    path = tmp_path / "parent.json"
    path.write_text("{}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "inference.pt"
    parent = SimpleNamespace(
        config_path=path,
        config_sha256=digest,
        inference_checkpoint=checkpoint,
        inference_sha256="a" * 64,
        full_resume_sha256="b" * 64,
        selected_epoch=1,
        plan_path=tmp_path / "plan.yaml",
    )
    model = SimpleNamespace(eval=lambda: "model")
    loaded = SimpleNamespace(
        model=model,
        source="source",
        activation=SimpleNamespace(quantizer_count=124),
        catalog=SimpleNamespace(summary=lambda: {"modules": 148}),
    )
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return loaded

    monkeypatch.setattr(LockedQATParentSpec, "from_yaml", lambda path: parent)
    monkeypatch.setattr(
        Full35QATRuntime,
        "from_yaml",
        lambda path: SimpleNamespace(load_deployment_parent=load),
    )
    plan = SimpleNamespace(
        locked_qat_parent=path,
        locked_qat_parent_sha256=digest,
        activation_checkpoint=checkpoint,
        activation_checkpoint_sha256="a" * 64,
        weight_study=SimpleNamespace(expected_catalog={"modules": 148}),
    )
    return plan, calls


def test_locked_ptq_retains_activation(tmp_path, monkeypatch):
    plan, calls = setup_parent(tmp_path, monkeypatch)
    model, source, activation, evidence = build_locked_parent(plan)
    assert (model, source, activation.quantizer_count) == ("model", "source", 124)
    assert evidence["activation_recalibrated"] is False
    assert calls[0][1] == {
        "checkpoint_sha256": "a" * 64,
        "full_resume_sha256": "b" * 64,
        "epoch": 1,
    }


def test_locked_ptq_rejects_manifest_drift(tmp_path, monkeypatch):
    plan, calls = setup_parent(tmp_path, monkeypatch)
    plan.locked_qat_parent.write_text('{"drift":true}')
    with pytest.raises(ValueError, match="manifest changed"):
        build_locked_parent(plan)
    assert calls == []


def test_locked_ptq_rejects_checkpoint_mismatch(tmp_path, monkeypatch):
    plan, calls = setup_parent(tmp_path, monkeypatch)
    plan.activation_checkpoint_sha256 = "c" * 64
    with pytest.raises(ValueError, match="checkpoint differs"):
        build_locked_parent(plan)
    assert calls == []


def test_locked_ptq_rejects_catalog_drift(tmp_path, monkeypatch):
    plan, _ = setup_parent(tmp_path, monkeypatch)
    plan.weight_study.expected_catalog = {"modules": 147}
    with pytest.raises(ValueError, match="catalog differs"):
        build_locked_parent(plan)


def test_partial_replacement_keeps_quantized_parent_cost():
    record = {
        "formats": [{"sites": [{"path": "a"}]}],
        "weight_elements": 8,
        "weight_code_bytes": 2,
        "scale_bytes": 4,
    }
    updated = add_inherited_cost(record, {"a": 12, "b": 8})
    assert updated["effective_quantized_modules"] == 2
    assert _packed_cost(deployment_weight_elements=16, policy_record=updated) == (
        14,
        64,
    )
