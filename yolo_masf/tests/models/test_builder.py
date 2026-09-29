from __future__ import annotations

from unittest.mock import patch

import torch
from torch import nn
from ultralytics.nn.modules import Detect

from masf_yolo.models.builder import P2_SLOT_INDEX, P3_SLOT_INDEX, build_model
from masf_yolo.models.mfam import MFAM, PartialMFAM


def test_b1_has_identity_slots_and_four_literal_strides() -> None:
    model = build_model("B1")

    assert isinstance(model.model[P2_SLOT_INDEX], nn.Identity)
    assert isinstance(model.model[P3_SLOT_INDEX], nn.Identity)
    assert isinstance(model.model[-1], Detect)
    assert model.model[-1].nl == 4
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]


def test_m0_slot_is_installed_before_constructor_stride_forward() -> None:
    seen: list[tuple[int, ...]] = []
    original = MFAM.forward

    def spy(module: MFAM, value: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(value.shape))
        return original(module, value)

    with patch.object(MFAM, "forward", spy):
        model = build_model("M0")

    assert seen
    assert seen[0][-2:] == (64, 64)
    assert isinstance(model.model[P2_SLOT_INDEX], MFAM)
    assert model.model[P2_SLOT_INDEX].kernels == (3, 5, 7, 9)


def test_partial_variants_use_exact_slots_without_global_registry_mutation() -> None:
    from ultralytics.nn import tasks

    before = set(tasks.__dict__)
    m2 = build_model("M2")
    m3 = build_model("M3")

    assert isinstance(m2.model[P2_SLOT_INDEX], PartialMFAM)
    assert isinstance(m3.model[P2_SLOT_INDEX], PartialMFAM)
    assert m2.model[P2_SLOT_INDEX].processed_ratio == 0.5
    assert m3.model[P2_SLOT_INDEX].processed_ratio == 0.25
    assert set(tasks.__dict__) == before


def test_m7_uses_full_channel_357_slot_and_four_scales() -> None:
    model = build_model("M7")

    assert isinstance(model.model[P2_SLOT_INDEX], MFAM)
    assert model.model[P2_SLOT_INDEX].channels == 128
    assert model.model[P2_SLOT_INDEX].kernels == (3, 5, 7)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert model.model[-1].nl == 4


def test_p3m_uses_full_mfam_only_in_p3_slot() -> None:
    model = build_model("P3M")

    assert isinstance(model.model[P2_SLOT_INDEX], nn.Identity)
    assert isinstance(model.model[P3_SLOT_INDEX], MFAM)
    p3_mfam = model.model[P3_SLOT_INDEX]
    assert p3_mfam.channels == 256
    assert p3_mfam.kernels == (3, 5, 7)
    assert tuple(p3_mfam.branches[2][0].conv.kernel_size) == (1, 7)
    assert tuple(p3_mfam.branches[2][1].conv.kernel_size) == (7, 1)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert model.model[-1].nl == 4


def test_detect_receives_p2_p3_p4_p5_feature_shapes() -> None:
    model = build_model("M1").eval()
    captured: list[list[tuple[int, ...]]] = []

    def hook(_module: nn.Module, args: tuple[object, ...]) -> None:
        captured.append([tuple(tensor.shape) for tensor in args[0]])

    handle = model.model[-1].register_forward_pre_hook(hook)
    with torch.no_grad():
        model(torch.zeros(1, 3, 256, 256))
    handle.remove()

    assert [shape[-2:] for shape in captured[-1]] == [(64, 64), (32, 32), (16, 16), (8, 8)]


def test_official_backbone_parameter_shapes_are_preserved() -> None:
    from ultralytics.nn.tasks import DetectionModel

    official = DetectionModel("yolo11m.yaml", nc=2, verbose=False)
    p2 = build_model("B1")

    for index in range(11):
        assert type(p2.model[index]) is type(official.model[index])
        assert [tuple(parameter.shape) for parameter in p2.model[index].parameters()] == [
            tuple(parameter.shape) for parameter in official.model[index].parameters()
        ]
