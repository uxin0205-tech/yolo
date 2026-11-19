from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

from achitechure_1.model import graft_p3_masf, inspect_yolo26_graph

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "inputs" / "parent" / "best.pt"


def test_graft_preserves_graph_index_and_every_parent_tensor() -> None:
    network = YOLO(str(PARENT)).model
    before = {name: value.detach().clone() for name, value in network.state_dict().items()}

    report = graft_p3_masf(network, "full35")

    graph = inspect_yolo26_graph(network)
    assert report.p3_index == 16
    assert graph.detect_inputs == (16, 19, 22)
    assert graph.strides == (8, 16, 32)
    assert graph.end2end is True
    assert report.preserved_tensors == len(before)
    after = network.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert any(name.startswith("model.16.p3_masf.") for name in after)


def test_partial75_uses_64_context_channels_at_real_p3_seam() -> None:
    network = YOLO(str(PARENT)).model

    graft_p3_masf(network, "partial75")

    module = network.model[16].p3_masf
    assert module.context_channels == 64
    assert module.bypass_channels == 192
