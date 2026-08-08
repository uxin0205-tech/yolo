from __future__ import annotations

import torch
from ultralytics.nn.tasks import DetectionModel

from masf_yolo.models.builder import build_model
from masf_yolo.models.transfer import transfer_b1_canonical, transfer_official_weights


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
