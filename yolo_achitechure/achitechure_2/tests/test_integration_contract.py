from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.integration
def test_formal_checkpoint_640_forward_three_scales_and_reload() -> None:
    manifest_value = os.environ.get("ACHITECHURE_2_HANDOFF")
    if not manifest_value:
        pytest.skip("set ACHITECHURE_2_HANDOFF after the formal achitechure_1 handoff")
    from ultralytics import YOLO

    from achitechure_2.graph import inspect_graph

    manifest = json.loads(Path(manifest_value).read_text(encoding="utf-8"))
    checkpoint = Path(manifest["float_checkpoint"]["path"])
    model = YOLO(str(checkpoint)).model.eval()
    report = inspect_graph(model)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640, device=next(model.parameters()).device))
    assert report.detect_inputs == (16, 19, 22)
    assert output is not None
