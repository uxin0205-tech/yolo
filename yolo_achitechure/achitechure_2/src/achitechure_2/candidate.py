"""Independent candidate construction and exact tensor transfer reporting."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import CANDIDATES, PROTECTED_C3K2_LAYERS, CandidateSpec
from .graph import GraphReport, assert_candidate_graph, inspect_c3k2_layer, inspect_graph, network_layers
from .lite_c3k2 import LiteC3k2


@dataclass(frozen=True)
class ShapeMismatch:
    name: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class TransferReport:
    seed: int
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[ShapeMismatch, ...]

    @property
    def loaded_count(self) -> int:
        return len(self.loaded)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matched"] = payload["loaded"]
        payload["matched_count"] = self.loaded_count
        payload["loaded_count"] = self.loaded_count
        return payload


@dataclass(frozen=True)
class CandidateBuild:
    candidate_id: str
    graph: GraphReport
    transfer: TransferReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "graph": self.graph.to_dict(),
            "transfer": self.transfer.to_dict(),
        }


def _construction_args(layer: nn.Module) -> tuple[int, int, int, bool, int]:
    report = inspect_c3k2_layer(-1, layer)
    blocks = [block for outer in layer.m for block in outer.m]
    shortcut = all(bool(getattr(block, "add", False)) for block in blocks)
    groups = int(getattr(getattr(blocks[0].cv2, "conv", None), "groups", 1))
    return report.input_channels, report.output_channels, report.outer_repeats, shortcut, groups


def _transfer(target: nn.Module, source_state: dict[str, torch.Tensor], seed: int) -> TransferReport:
    target_state = target.state_dict()
    loaded = tuple(
        sorted(
            name
            for name, value in source_state.items()
            if name in target_state and value.shape == target_state[name].shape
        )
    )
    shape_mismatch = tuple(
        ShapeMismatch(name, tuple(source_state[name].shape), tuple(target_state[name].shape))
        for name in sorted(set(source_state) & set(target_state))
        if source_state[name].shape != target_state[name].shape
    )
    missing = tuple(sorted(set(target_state) - set(source_state)))
    unexpected = tuple(sorted(set(source_state) - set(target_state)))
    compatible = {name: source_state[name] for name in loaded}
    target.load_state_dict(compatible, strict=False)
    return TransferReport(seed, loaded, missing, unexpected, shape_mismatch)


def build_candidate(
    parent: nn.Module,
    candidate_id: str,
    *,
    seed: int = 0,
) -> tuple[nn.Module, CandidateBuild]:
    """Build one candidate from an immutable parent, never from another candidate."""

    candidate_id = candidate_id.upper()
    if candidate_id not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate_id}; choose from {sorted(CANDIDATES)}")
    spec: CandidateSpec = CANDIDATES[candidate_id]
    parent_graph = inspect_graph(parent)
    parent_state = {name: value.detach().clone() for name, value in parent.state_dict().items()}
    model = copy.deepcopy(parent)
    layers = network_layers(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for index in spec.target_layers:
            c1, c2, outer_n, shortcut, groups = _construction_args(layers[index])
            replacement = LiteC3k2(
                c1,
                c2,
                outer_n,
                config=spec.lite,
                shortcut=shortcut,
                groups=groups,
            ).to(device=next(layers[index].parameters()).device, dtype=next(layers[index].parameters()).dtype)
            for attribute in ("i", "f", "type", "np"):
                if hasattr(layers[index], attribute):
                    setattr(replacement, attribute, getattr(layers[index], attribute))
            layers[index] = replacement
    transfer = _transfer(model, parent_state, seed)
    graph = assert_candidate_graph(model, candidate_id, spec.target_layers)
    if (
        graph.masf_variant != parent_graph.masf_variant
        or graph.attention_paths != parent_graph.attention_paths
    ):
        raise AssertionError("candidate construction changed a frozen inherited module")
    current = parent.state_dict()
    if any(not torch.equal(value, current[name]) for name, value in parent_state.items()):
        raise AssertionError("candidate construction mutated the parent")
    for index in PROTECTED_C3K2_LAYERS:
        if type(layers[index]) is not type(network_layers(parent)[index]):
            raise AssertionError(f"protected layer {index} changed class")
    return model, CandidateBuild(candidate_id, graph, transfer)


def graft_pose_candidate(
    detect_candidate: nn.Module,
    candidate_id: str,
    *,
    data_yaml: str | Path,
    seed: int = 0,
) -> tuple[nn.Module, CandidateBuild]:
    """Graft the validated candidate trunk into the official local YOLO26m Pose26 graph."""

    from ultralytics.nn.tasks import PoseModel

    candidate_id = candidate_id.upper()
    if candidate_id not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate_id}")
    spec = CANDIDATES[candidate_id]
    source_graph = assert_candidate_graph(detect_candidate, candidate_id, spec.target_layers)
    if source_graph.task != "detect":
        raise TypeError(f"Pose graft requires a Detect candidate, got {source_graph.task}")
    dataset = __import__("yaml").safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or dataset.get("kpt_shape") != [2, 3]:
        raise ValueError("Pose dataset must declare kpt_shape [2, 3]")
    names = dataset.get("names")
    nc = int(dataset.get("nc", len(names) if isinstance(names, (list, dict)) else 0))
    if nc <= 0:
        raise ValueError("Pose dataset must declare nc/names")

    source_state = {name: value.detach().clone() for name, value in detect_candidate.state_dict().items()}
    source_layers = network_layers(detect_candidate)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        pose = PoseModel(
            "yolo26m-pose.yaml",
            nc=nc,
            data_kpt_shape=tuple(dataset["kpt_shape"]),
            verbose=False,
        )
    pose_layers = network_layers(pose)
    for index in range(23):
        pose_layers[index] = copy.deepcopy(source_layers[index])
    first_parameter = next(detect_candidate.parameters())
    pose = pose.to(device=first_parameter.device, dtype=first_parameter.dtype)
    pose.names = names
    transfer = _transfer(pose, source_state, seed)
    graph = assert_candidate_graph(pose, candidate_id, spec.target_layers)
    if graph.task != "pose" or graph.head_type != "Pose26":
        raise AssertionError(f"official Pose26 graft failed: {graph.head_type}")
    if graph.masf_variant != source_graph.masf_variant:
        raise AssertionError("Pose graft changed inherited MASF")
    current = detect_candidate.state_dict()
    if any(not torch.equal(value, current[name]) for name, value in source_state.items()):
        raise AssertionError("Pose graft mutated the Detect candidate")
    return pose, CandidateBuild(candidate_id, graph, transfer)


def write_build_report(report: CandidateBuild, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
