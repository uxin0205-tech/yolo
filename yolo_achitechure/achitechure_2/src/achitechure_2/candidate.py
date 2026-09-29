"""從不可變 Fusion Winner 建立單區域、單因子的 C0–C3。"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .config import CANDIDATES, SPEC_PATH, SPEC_VERSION, CandidateSpec, file_sha256
from .graph import inspect_c3k2_layer
from .intake import FUSION_KINDS, CandidateRegion
from .lite_c3k2 import LiteC3k2


@dataclass(frozen=True)
class ResolvedCandidate:
    """一個 base factor 與一個 handoff region 的唯一組合。"""

    base_candidate_id: str
    resolved_id: str
    fusion_kind: str
    region: CandidateRegion | None

    @property
    def spec(self) -> CandidateSpec:
        return CANDIDATES[self.base_candidate_id]


@dataclass(frozen=True)
class ShapeMismatch:
    name: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class TransferReport:
    seed: int
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[ShapeMismatch, ...]

    @property
    def loaded(self) -> tuple[str, ...]:
        return self.matched

    @property
    def loaded_count(self) -> int:
        return len(self.matched)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loaded"] = payload["matched"]
        payload["loaded_count"] = self.loaded_count
        payload["matched_count"] = self.loaded_count
        return payload


@dataclass(frozen=True)
class CandidateBuild:
    candidate_id: str
    resolved_id: str
    fusion_kind: str
    region_id: str | None
    region_role: str | None
    changed_fields: tuple[str, ...]
    changed_module_paths: tuple[str, ...]
    module_contracts: tuple[dict[str, Any], ...]
    transfer: TransferReport
    parent_unchanged: bool
    model_contract_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["transfer"] = self.transfer.to_dict()
        return payload


_ROLE_ORDER = {"shared": 0, "detect_specific": 1, "pose_specific": 2}
_PREFIX = {"shared": "S", "detect_specific": "D", "pose_specific": "P"}


def resolve_candidate_matrix(
    fusion_kind: str,
    regions: tuple[CandidateRegion, ...],
) -> tuple[ResolvedCandidate, ...]:
    """依 winner fusion kind 解析第一輪實際候選 IDs。"""

    if fusion_kind not in FUSION_KINDS:
        raise ValueError(f"不支援的 fusion_kind：{fusion_kind}")
    if not regions:
        raise ValueError("handoff 沒有 Candidate Region")
    roles = {region.role for region in regions}
    if fusion_kind == "shared_dual_head" and roles != {"shared"}:
        raise ValueError("shared_dual_head 只能解析 shared region")
    if fusion_kind == "routed_dual" and roles != {"detect_specific", "pose_specific"}:
        raise ValueError("routed_dual 必須同時解析 detect/pose task-specific regions")
    if fusion_kind == "partial_shared" and "shared" not in roles:
        raise ValueError("partial_shared 至少需要一個 shared region")
    if len({region.region_id for region in regions}) != len(regions):
        raise ValueError("Candidate Region IDs 不得重複")

    matrix: list[ResolvedCandidate] = [
        ResolvedCandidate("C0", "C0", fusion_kind, None)
    ]
    ordered = sorted(regions, key=lambda region: (_ROLE_ORDER[region.role], region.region_id))
    for region in ordered:
        for base_id in ("C1", "C2", "C3"):
            if fusion_kind == "shared_dual_head" and region.role == "shared":
                resolved_id = base_id
            else:
                resolved_id = f"{_PREFIX[region.role]}-{base_id}"
            matrix.append(ResolvedCandidate(base_id, resolved_id, fusion_kind, region))
    return tuple(matrix)


def _construction_args(layer: nn.Module) -> tuple[int, int, int, bool, int]:
    report = inspect_c3k2_layer(-1, layer)
    outer_blocks = tuple(getattr(layer, "m", ()))
    blocks = tuple(block for outer in outer_blocks for block in getattr(outer, "m", ()))
    if not blocks:
        raise TypeError(f"{type(layer).__name__} 沒有可重建的 inner bottleneck")
    shortcut = all(bool(getattr(block, "add", False)) for block in blocks)
    groups = int(getattr(getattr(blocks[0].cv2, "conv", None), "groups", 1))
    return (
        report.input_channels,
        report.output_channels,
        report.outer_repeats,
        shortcut,
        groups,
    )


def _replace_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, leaf = path.rpartition(".")
    parent = model.get_submodule(parent_path) if separator else model
    if leaf not in parent._modules:
        raise ValueError(f"module path 無法替換：{path}")
    parent._modules[leaf] = replacement


def _transfer(
    target: nn.Module,
    source_state: dict[str, torch.Tensor],
    *,
    seed: int,
) -> TransferReport:
    target_state = target.state_dict()
    matched = tuple(
        sorted(
            name
            for name, source in source_state.items()
            if name in target_state and source.shape == target_state[name].shape
        )
    )
    mismatch = tuple(
        ShapeMismatch(name, tuple(source_state[name].shape), tuple(target_state[name].shape))
        for name in sorted(set(source_state) & set(target_state))
        if source_state[name].shape != target_state[name].shape
    )
    missing = tuple(sorted(set(target_state) - set(source_state)))
    unexpected = tuple(sorted(set(source_state) - set(target_state)))
    target.load_state_dict({name: source_state[name] for name in matched}, strict=False)
    return TransferReport(seed, matched, missing, unexpected, mismatch)


def _contract(model: nn.Module) -> Any:
    method = getattr(model, "contract", None)
    return copy.deepcopy(method()) if callable(method) else None


def _path_owns_tensor(path: str, tensor_name: str) -> bool:
    return tensor_name == path or tensor_name.startswith(path + ".")


def build_candidate(
    parent: nn.Module,
    candidate: ResolvedCandidate | str,
    *,
    region: CandidateRegion | None = None,
    fusion_kind: str = "shared_dual_head",
    seed: int = 0,
) -> tuple[nn.Module, CandidateBuild]:
    """從同一 parent 深拷貝建立候選；非 C0 必須帶入 handoff region。"""

    if isinstance(candidate, str):
        base_id = candidate.upper()
        if base_id == "C0":
            resolved = ResolvedCandidate("C0", "C0", fusion_kind, None)
        else:
            if region is None:
                raise ValueError("v2 非 C0 候選必須提供 handoff Candidate Region")
            resolved = ResolvedCandidate(base_id, base_id, fusion_kind, region)
    else:
        resolved = candidate
    if resolved.base_candidate_id not in CANDIDATES:
        raise ValueError(f"未知候選：{resolved.base_candidate_id}")
    if resolved.base_candidate_id == "C0" and resolved.region is not None:
        raise ValueError("C0-Handoff 不得指定修改 region")
    if resolved.base_candidate_id != "C0" and resolved.region is None:
        raise ValueError("非 C0 候選缺少 Candidate Region")
    if seed < 0:
        raise ValueError("seed 必須為非負整數")

    spec = resolved.spec
    parent_state = {
        name: value.detach().clone() for name, value in parent.state_dict().items()
    }
    parent_contract = _contract(parent)
    model = copy.deepcopy(parent)
    changed_paths: tuple[str, ...] = ()
    if resolved.region is not None:
        changed_paths = resolved.region.module_paths
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for path in changed_paths:
                source_layer = model.get_submodule(path)
                baseline = inspect_c3k2_layer(-1, source_layer)
                if (
                    baseline.e != 0.5
                    or baseline.inner_n != 2
                    or baseline.kernel_mode != "3x3_3x3"
                    or baseline.use_rep
                ):
                    raise ValueError(f"{path}: handoff C3k2 baseline factors 不符合 C0")
                c1, c2, outer_n, shortcut, groups = _construction_args(source_layer)
                first_parameter = next(source_layer.parameters())
                replacement = LiteC3k2(
                    c1,
                    c2,
                    outer_n,
                    config=spec.lite,
                    shortcut=shortcut,
                    groups=groups,
                ).to(device=first_parameter.device, dtype=first_parameter.dtype)
                replacement.train(source_layer.training)
                for attribute in ("i", "f", "type", "np"):
                    if hasattr(source_layer, attribute):
                        setattr(replacement, attribute, getattr(source_layer, attribute))
                _replace_submodule(model, path, replacement)

    transfer = _transfer(model, parent_state, seed=seed)
    changed_prefixes = changed_paths
    illegal_mismatch = [
        item.name
        for item in transfer.shape_mismatch
        if not any(_path_owns_tensor(path, item.name) for path in changed_prefixes)
    ]
    illegal_missing = [
        name
        for name in (*transfer.missing, *transfer.unexpected)
        if not any(_path_owns_tensor(path, name) for path in changed_prefixes)
    ]
    if illegal_mismatch or illegal_missing:
        raise AssertionError(
            "候選修改逸出 Candidate Region："
            f"shape={illegal_mismatch[:5]} keys={illegal_missing[:5]}"
        )

    current_parent = parent.state_dict()
    parent_unchanged = current_parent.keys() == parent_state.keys() and all(
        torch.equal(value, current_parent[name]) for name, value in parent_state.items()
    )
    if not parent_unchanged:
        raise AssertionError("候選建構修改了 parent")
    model_contract_unchanged = _contract(model) == parent_contract
    if not model_contract_unchanged:
        raise AssertionError("候選建構修改了 Detect/Pose 模型契約")
    if resolved.base_candidate_id == "C0":
        if transfer.shape_mismatch or transfer.missing or transfer.unexpected:
            raise AssertionError("C0-Handoff state_dict 不等價")
        if any(
            not torch.equal(value, model.state_dict()[name])
            for name, value in parent_state.items()
        ):
            raise AssertionError("C0-Handoff tensor 不等價")

    contracts: list[dict[str, Any]] = []
    for path in changed_paths:
        item = inspect_c3k2_layer(-1, model.get_submodule(path))
        contracts.append(
            {
                "path": path,
                "class_name": item.class_name,
                "input_channels": item.input_channels,
                "output_channels": item.output_channels,
                "hidden_channels": item.hidden_channels,
                "outer_repeats": item.outer_repeats,
                "e": item.e,
                "inner_n": item.inner_n,
                "kernel_mode": item.kernel_mode,
                "use_rep": item.use_rep,
            }
        )
    report = CandidateBuild(
        candidate_id=resolved.base_candidate_id,
        resolved_id=resolved.resolved_id,
        fusion_kind=resolved.fusion_kind,
        region_id=resolved.region.region_id if resolved.region else None,
        region_role=resolved.region.role if resolved.region else None,
        changed_fields=spec.changed_fields,
        changed_module_paths=changed_paths,
        module_contracts=tuple(contracts),
        transfer=transfer,
        parent_unchanged=parent_unchanged,
        model_contract_unchanged=model_contract_unchanged,
    )
    return model, report


def graph_snapshot(model: nn.Module, report: CandidateBuild) -> dict[str, Any]:
    """建立 review-only snapshot；不可直接交給 YOLO(yaml)。"""

    contract = _contract(model)
    return {
        "schema_version": 2,
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "standalone_loadable": False,
        "builder": "achitechure_2",
        "base_candidate": report.candidate_id,
        "resolved_candidate": report.resolved_id,
        "fusion_kind": report.fusion_kind,
        "candidate_region": {
            "region_id": report.region_id,
            "role": report.region_role,
            "module_paths": list(report.changed_module_paths),
        },
        "model_contract": contract,
        "modified_modules": list(report.module_contracts),
        "transfer_summary": {
            "matched": len(report.transfer.matched),
            "missing": len(report.transfer.missing),
            "unexpected": len(report.transfer.unexpected),
            "shape_mismatch": len(report.transfer.shape_mismatch),
        },
    }


def write_graph_snapshot(
    model: nn.Module,
    report: CandidateBuild,
    destination: str | Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(graph_snapshot(model, report), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def write_build_report(report: CandidateBuild, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def graft_pose_candidate(*_: Any, **__: Any) -> None:
    raise RuntimeError(
        "v2 不再 graft 獨立 Pose graph；Pose head 已由 yolo_combine winner handoff 提供"
    )
