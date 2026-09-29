"""Fixed-shape SP2 ONNX export and hardware-operator audit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import onnx

from masf_yolo.artifacts.io import atomic_write_json
from masf_yolo.contracts import sha256_file
from masf_yolo.models.selective import SelectiveP2Detect


FORBIDDEN_DATA_OPERATORS = frozenset({"TopK", "GatherElements", "NonMaxSuppression"})


def inspect_onnx(path: Path) -> dict[str, Any]:
    """Validate static output and reject data-dependent selection operators."""
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    operators = Counter(node.op_type for node in graph.graph.node)
    forbidden = sorted(operator for operator in FORBIDDEN_DATA_OPERATORS if operators[operator])
    if forbidden:
        raise ValueError(f"SP2 ONNX contains forbidden operators: {forbidden}")
    producers = {output: node.op_type for node in graph.graph.node for output in node.output}
    shape_gathers: list[str] = []
    for node in graph.graph.node:
        if node.op_type != "Gather":
            continue
        if not node.input or producers.get(node.input[0]) != "Shape":
            raise ValueError(f"SP2 ONNX contains data-dependent Gather: {node.name}")
        shape_gathers.append(node.name)
    output_shapes = [
        [dimension.dim_value for dimension in output.type.tensor_type.shape.dim]
        for output in graph.graph.output
    ]
    if len(output_shapes) != 1 or output_shapes[0][:2] != [1, 6]:
        raise ValueError(f"unexpected SP2 ONNX output shape: {output_shapes}")
    return {
        "ir_version": graph.ir_version,
        "opsets": {item.domain or "default": item.version for item in graph.opset_import},
        "output_shapes": output_shapes,
        "operators": dict(sorted(operators.items())),
        "shape_only_gathers": shape_gathers,
        "forbidden_operators": forbidden,
    }


def export_sp2_onnx(checkpoint: Path, output_dir: Path, *, imgsz: int = 640) -> dict[str, Any]:
    """Export the trained native SP2 checkpoint through the pinned Ultralytics exporter."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = YOLO(str(checkpoint.resolve()), task="detect")
    model = wrapper.model.eval().cpu()
    if not isinstance(model.model[-1], SelectiveP2Detect):
        raise TypeError("SP2 export checkpoint does not contain SelectiveP2Detect")
    if not isinstance(model.args, dict):
        model.args = vars(model.args).copy()
    model.args["imgsz"] = imgsz
    model.task = "detect"
    model.pt_path = str((output_dir / "sp2.pt").resolve())
    wrapper.model = model
    exported = Path(
        wrapper.export(
            format="onnx",
            imgsz=imgsz,
            batch=1,
            device="cpu",
            dynamic=False,
            simplify=False,
            opset=17,
            verbose=False,
        )
    ).resolve()
    expected = (output_dir / "sp2.onnx").resolve()
    if exported != expected:
        raise RuntimeError(f"SP2 exporter wrote unexpected path: {exported}")
    inspection = inspect_onnx(expected)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(expected),
        "onnx_hash": sha256_file(expected),
        "imgsz": imgsz,
        "batch": 1,
        "dynamic": False,
        "opset": 17,
        "data_exposed": True,
        "inspection": inspection,
    }
    atomic_write_json(output_dir / "manifest.json", result)
    return result
