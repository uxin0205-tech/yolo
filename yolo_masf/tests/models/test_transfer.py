from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from masf_yolo.models.builder import build_b1r_model, build_model, build_p3_model
import masf_yolo.models.transfer as transfer_module
from masf_yolo.models.transfer import transfer_b1_canonical, transfer_official_weights, transfer_same_graph_compatible


def _fill_module(model: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(value)
            else:
                tensor.zero_()


def test_official_transfer_maps_backbone_shared_neck_and_p3_regression() -> None:
    source = DetectionModel("yolo11m.yaml", nc=2, verbose=False)
    destination = build_model("B1")
    _fill_module(source, 0.25)

    report = transfer_official_weights(destination, source)
    state = destination.state_dict()

    assert torch.all(state["model.0.conv.weight"] == 0.25)
    assert torch.all(state["model.16.cv1.conv.weight"] == 0.25)
    assert torch.all(state["model.25.conv.weight"] == 0.25)
    assert torch.all(state["model.31.cv2.1.0.conv.weight"] == 0.25)
    assert "model.19.cv1.conv.weight" in report.missing
    assert "model.31.cv2.0.0.conv.weight" in report.missing
    assert any(key.startswith("model.31.cv3.1") for key in report.shape_mismatch)
    assert report.matched
    assert report.unexpected == ()


def test_official_transfer_report_has_one_disposition_per_destination_tensor() -> None:
    source = DetectionModel("yolo11m.yaml", nc=2, verbose=False)
    destination = build_model("B1")

    source_state = dict(source.state_dict())
    source_state["model.999.extra"] = torch.ones(1)
    report = transfer_official_weights(destination, source_state)

    dispositions = set(report.matched) | set(report.missing) | set(report.shape_mismatch)
    assert dispositions == set(destination.state_dict())
    assert not (set(report.matched) & set(report.missing))
    assert report.unexpected == ("model.999.extra",)


def test_clean_same_graph_transfer_keeps_coco_class_outputs_new() -> None:
    source = DetectionModel("yolo11m.yaml", nc=80, verbose=False)
    destination = build_p3_model()
    _fill_module(source, 0.25)
    before = destination.state_dict()["model.23.cv3.0.2.bias"].clone()
    report = transfer_same_graph_compatible(destination, source)

    assert torch.all(destination.state_dict()["model.0.conv.weight"] == 0.25)
    torch.testing.assert_close(destination.state_dict()["model.23.cv3.0.2.bias"], before)
    assert "model.23.cv3.0.2.bias" in report.shape_mismatch
    assert not report.missing


def test_requested_pose_derived_initializer_transfers_all_compatible_b1_tensors() -> None:
    model = build_model(
        "B1",
        source_weights=Path("bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt"),
    )
    report = model.masf_transfer_report

    assert len(report["matched"]) == 601
    assert len(report["missing"]) == 154
    assert len(report["shape_mismatch"]) == 48
    assert not report["unexpected"]
    assert all(key.startswith("model.31.cv3.") for key in report["shape_mismatch"])
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]


def test_b1r_transfer_preserves_b0_p3_p5_classification_shapes() -> None:
    source = DetectionModel("yolo11m.yaml", nc=2, verbose=False)
    destination = build_b1r_model()
    _fill_module(source, 0.25)
    report = transfer_official_weights(destination, source)
    assert not report.shape_mismatch
    assert len(report.matched) == 649
    assert len(report.missing) == 154
    assert report.unexpected == ()
    assert torch.all(destination.state_dict()["model.31.cv3.1.1.1.conv.weight"] == 0.25)


def test_p3_variant_leaves_only_the_replaced_p3_slot_new():
    source = DetectionModel("yolo11m.yaml", nc=2, verbose=False)
    destination = build_p3_model("PaperFormula-Full")
    _fill_module(source, 0.25)
    report = transfer_module.transfer_b0_p3_parent(destination, source)
    assert report.shape_mismatch == {}
    assert report.unexpected == ()
    assert report.missing
    assert all(key.startswith("model.16.") for key in report.missing)


def test_variant_transfer_loads_all_b1_tensors_but_keeps_mfam_random() -> None:
    b1 = build_model("B1")
    m0 = build_model("M0")
    _fill_module(b1, 0.75)
    before = m0.state_dict()["model.20.branches.0.conv.weight"].clone()

    report = transfer_b1_canonical(m0, b1.state_dict())

    assert torch.all(m0.state_dict()["model.0.conv.weight"] == 0.75)
    torch.testing.assert_close(m0.state_dict()["model.20.branches.0.conv.weight"], before)
    assert "model.20.branches.0.conv.weight" in report.missing
    assert not report.shape_mismatch


def test_m7_transfer_keeps_every_mfam_tensor_random() -> None:
    b1 = build_model("B1")
    m7 = build_model("M7")
    _fill_module(b1, 0.625)
    before = {
        key: tensor.clone()
        for key, tensor in m7.state_dict().items()
        if key.startswith("model.20.")
    }

    report = transfer_b1_canonical(m7, b1.state_dict())

    assert torch.all(m7.state_dict()["model.0.conv.weight"] == 0.625)
    assert before
    for key, tensor in before.items():
        torch.testing.assert_close(m7.state_dict()[key], tensor)
        assert key in report.missing
    assert not report.shape_mismatch


def test_sp2p_transfer_uses_sp2_for_shared_state_and_partial_only_for_slot() -> None:
    assert hasattr(transfer_module, "transfer_sp2p_parents")
    sp2 = build_model("SP2")
    m3 = build_model("M3")
    target = build_model("SP2M3")
    shared_key = "model.0.conv.weight"
    slot_key = "model.20.process.branches.0.conv.weight"
    with torch.no_grad():
        sp2.state_dict()[shared_key].fill_(0.25)
        m3.state_dict()[slot_key].fill_(0.75)
        m3.state_dict()[shared_key].fill_(0.99)

    report = transfer_module.transfer_sp2p_parents(
        target,
        sp2.state_dict(),
        m3.state_dict(),
        selected_partial="M3",
    )

    assert torch.all(target.state_dict()[shared_key] == 0.25)
    assert torch.all(target.state_dict()[slot_key] == 0.75)
    assert slot_key in report.partial_keys
    assert all(key.startswith("model.20.") for key in report.partial_keys)
    assert all(not key.startswith("model.20.") for key in report.shared_keys)
    assert report.selected_partial == "M3"


def test_sp2p_transfer_rejects_wrong_parent_missing_key_and_shape() -> None:
    assert hasattr(transfer_module, "transfer_sp2p_parents")
    sp2 = build_model("SP2")
    m3 = build_model("M3")

    with pytest.raises(ValueError, match="target.*M2"):
        transfer_module.transfer_sp2p_parents(
            build_model("SP2M3"), sp2.state_dict(), m3.state_dict(), selected_partial="M2"
        )

    missing = dict(m3.state_dict())
    missing.pop("model.20.process.branches.0.conv.weight")
    with pytest.raises(ValueError, match="slot keys"):
        transfer_module.transfer_sp2p_parents(
            build_model("SP2M3"), sp2.state_dict(), missing, selected_partial="M3"
        )

    wrong_shape = dict(m3.state_dict())
    wrong_shape["model.20.process.branches.0.conv.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        transfer_module.transfer_sp2p_parents(
            build_model("SP2M3"), sp2.state_dict(), wrong_shape, selected_partial="M3"
        )
