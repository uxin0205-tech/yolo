"""由 EXPERIMENT_SPEC.md 衍生的 fail-closed 設定契約。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .lite_c3k2 import LiteC3k2Config

SPEC_VERSION = "2.0.2"
EXPECTED_ULTRALYTICS_VERSION = "8.4.90"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "EXPERIMENT_SPEC.md"
CANDIDATE_DIR = PROJECT_ROOT / "configs/candidates"
RUNTIME_OVERRIDE_ALLOWLIST = frozenset({"name", "project", "device", "workers", "cache"})
FORBIDDEN_COMBINATIONS = ("C1+C2", "C1+C3", "C2+C3")
BASELINE_FACTORS = (0.5, 2, "3x3_3x3", False)
TRAINING_FILES = (
    "configs/training/cpu-smoke.yaml",
    "configs/training/float-extension.yaml",
    "configs/training/float-main.yaml",
    "configs/training/quant-qat.yaml",
)

# 舊版程式可能匯入這些名稱；v2 刻意不賦予固定 layer index。
TARGET_LAYERS: tuple[int, ...] = ()
P5_ONLY_LAYER: tuple[int, ...] = ()
PROTECTED_C3K2_LAYERS: tuple[int, ...] = ()
DETECT_INPUTS = (16, 19, 22)
STRIDES = (8, 16, 32)
ATTENTION_PATHS: tuple[str, ...] = ()
PERMANENT_FROZEN_MODULES: tuple[str, ...] = ()


class ConfigContractError(ValueError):
    """設定無法證明符合唯一正式規格。"""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(path: str | Path) -> str:
    return file_sha256(path)


def composite_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = Path(path).resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigContractError(f"{resolved}: YAML 根節點必須是 mapping")
    return payload


def validate_spec_metadata(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    spec_path: str | Path = SPEC_PATH,
) -> None:
    expected_sha = file_sha256(spec_path)
    if payload.get("spec_version") != SPEC_VERSION:
        raise ConfigContractError(
            f"{path}: spec_version={payload.get('spec_version')!r}，預期 {SPEC_VERSION!r}"
        )
    if payload.get("spec_sha256") != expected_sha:
        raise ConfigContractError(f"{path}: spec_sha256 與 EXPERIMENT_SPEC.md 不符")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    factor_name: str
    factors: tuple[float, int, str, bool]
    baseline_factors: tuple[float, int, str, bool]
    changed_fields: tuple[str, ...]
    target_source: str
    hardcoded_layer_indices: bool
    scope_resolution: Mapping[str, tuple[str, ...]]
    config_path: Path
    quantization_eligible: bool | str
    lite: LiteC3k2Config | None

    @property
    def declared_factors(self) -> tuple[float, int, str, bool]:
        return self.factors

    @property
    def target_layers(self) -> tuple[int, ...]:
        """v2 不允許候選 YAML 預先宣告實際 layer index。"""

        return ()


_EXPECTED_FACTORS: dict[str, tuple[tuple[float, int, str, bool], tuple[str, ...], str]] = {
    "C0": (BASELINE_FACTORS, (), "none"),
    "C1": ((0.375, 2, "3x3_3x3", False), ("e",), "hidden_expansion"),
    "C2": ((0.5, 1, "3x3_3x3", False), ("inner_n",), "inner_depth"),
    "C3": ((0.5, 2, "1x1_3x3", False), ("kernel_mode",), "first_inner_kernel"),
}

_CANDIDATE_KEYS = {
    "schema_version",
    "spec_version",
    "spec_sha256",
    "kind",
    "candidate_id",
    "title_zh",
    "summary_zh",
    "status",
    "model_scale",
    "task_contract",
    "parent_policy",
    "target_policy",
    "scope_resolution",
    "factors",
    "baseline_factors",
    "changed_fields",
    "factor_name",
    "protected_policy",
    "forbidden_combinations",
    "quantization_eligible",
    "standalone_loadable",
    "builder",
    "architecture_delta",
}


def _factor_tuple(value: Any, path: Path, label: str) -> tuple[float, int, str, bool]:
    if not isinstance(value, dict) or set(value) != {"e", "inner_n", "kernel_mode", "use_rep"}:
        raise ConfigContractError(f"{path}: {label} 必須恰好包含 e/inner_n/kernel_mode/use_rep")
    if (
        not isinstance(value["e"], (int, float))
        or isinstance(value["e"], bool)
        or not isinstance(value["inner_n"], int)
        or isinstance(value["inner_n"], bool)
        or value["kernel_mode"] not in {"3x3_3x3", "1x1_3x3"}
        or not isinstance(value["use_rep"], bool)
    ):
        raise ConfigContractError(f"{path}: {label} 的型別或值不合法")
    return (
        float(value["e"]),
        int(value["inner_n"]),
        str(value["kernel_mode"]),
        bool(value["use_rep"]),
    )


def _candidate_from_yaml(path: Path, *, spec_path: Path) -> CandidateSpec:
    payload = load_yaml(path)
    validate_spec_metadata(payload, path, spec_path=spec_path)
    if set(payload) != _CANDIDATE_KEYS:
        raise ConfigContractError(
            f"{path}: 欄位不完整，missing={sorted(_CANDIDATE_KEYS - set(payload))} "
            f"unknown={sorted(set(payload) - _CANDIDATE_KEYS)}"
        )
    if (
        payload["schema_version"] != 3
        or payload["kind"] != "architecture_candidate"
        or payload["status"] != "required"
        or payload["model_scale"] != "m"
        or payload["standalone_loadable"] is not False
        or payload["builder"] != "achitechure_2"
    ):
        raise ConfigContractError(f"{path}: 候選固定 metadata 漂移")
    if payload["task_contract"] != {
        "detect_nc": 80,
        "pose_nc": 2,
        "pose_kpt_shape": [2, 3],
    }:
        raise ConfigContractError(f"{path}: Detect/Pose 主任務契約漂移")
    if payload["parent_policy"] != {
        "source": "accepted_yolo_combine_winner",
        "checkpoint_role": "c0_handoff",
        "independent_build": True,
        "seed": 0,
    }:
        raise ConfigContractError(f"{path}: parent policy 漂移")
    target = payload["target_policy"]
    if target != {
        "source": "handoff.candidate_regions",
        "module_type": "C3k2-compatible",
        "hardcoded_layer_indices": False,
    }:
        raise ConfigContractError(f"{path}: 實際 target 必須由 handoff 解析，禁止固定 layer index")
    expected_scopes = {
        "shared_dual_head": ["shared"],
        "routed_dual": ["detect_specific", "pose_specific"],
        "partial_shared": ["shared", "detect_specific", "pose_specific"],
    }
    if payload["scope_resolution"] != expected_scopes:
        raise ConfigContractError(f"{path}: fusion-kind scope resolution 漂移")
    if tuple(payload["forbidden_combinations"]) != FORBIDDEN_COMBINATIONS:
        raise ConfigContractError(f"{path}: 禁止組合規則漂移")

    candidate_id = str(payload["candidate_id"])
    if candidate_id not in _EXPECTED_FACTORS:
        raise ConfigContractError(f"{path}: 第一輪只允許 C0/C1/C2/C3")
    factors = _factor_tuple(payload["factors"], path, "factors")
    baseline = _factor_tuple(payload["baseline_factors"], path, "baseline_factors")
    if baseline != BASELINE_FACTORS:
        raise ConfigContractError(f"{path}: baseline factors 漂移")
    actual_changed = tuple(
        name
        for name, actual, original in zip(
            ("e", "inner_n", "kernel_mode", "use_rep"), factors, baseline, strict=True
        )
        if actual != original
    )
    declared_changed = tuple(str(item) for item in payload["changed_fields"])
    if actual_changed != declared_changed or len(declared_changed) > 1:
        raise ConfigContractError(f"{path}: 每個候選只能改變一個 factor")
    expected_factors, expected_changed, expected_name = _EXPECTED_FACTORS[candidate_id]
    if (
        factors != expected_factors
        or declared_changed != expected_changed
        or payload["factor_name"] != expected_name
    ):
        raise ConfigContractError(f"{path}: {candidate_id} 候選矩陣漂移")
    expected_eligibility: bool | str = True if candidate_id == "C0" else "pending_user_decision"
    if payload["quantization_eligible"] != expected_eligibility:
        raise ConfigContractError(f"{path}: quantization eligibility 不得預先替使用者決定")
    lite = None
    if candidate_id != "C0":
        lite = LiteC3k2Config(
            e=factors[0],
            inner_n=factors[1],
            kernel_mode=factors[2],
            use_rep=factors[3],
        )
    scopes = {
        str(kind): tuple(str(role) for role in roles)
        for kind, roles in payload["scope_resolution"].items()
    }
    return CandidateSpec(
        candidate_id=candidate_id,
        factor_name=str(payload["factor_name"]),
        factors=factors,
        baseline_factors=baseline,
        changed_fields=declared_changed,
        target_source=str(target["source"]),
        hardcoded_layer_indices=bool(target["hardcoded_layer_indices"]),
        scope_resolution=scopes,
        config_path=path.resolve(),
        quantization_eligible=payload["quantization_eligible"],
        lite=lite,
    )


def load_candidate_specs(
    config_dir: str | Path = CANDIDATE_DIR,
    *,
    spec_path: str | Path = SPEC_PATH,
) -> dict[str, CandidateSpec]:
    paths = sorted(Path(config_dir).glob("*.yaml"))
    specs = [_candidate_from_yaml(path, spec_path=Path(spec_path)) for path in paths]
    candidates = {candidate.candidate_id: candidate for candidate in specs}
    if len(candidates) != len(specs):
        raise ConfigContractError("候選 candidate_id 重複")
    if tuple(candidates) != ("C0", "C1", "C2", "C3"):
        raise ConfigContractError(
            f"第一輪候選矩陣漂移：實際 {tuple(candidates)}，預期 ('C0', 'C1', 'C2', 'C3')"
        )
    return candidates


CANDIDATES = load_candidate_specs()

_TRAINING_KEYS = {
    "schema_version",
    "spec_version",
    "spec_sha256",
    "kind",
    "config_id",
    "title_zh",
    "summary_zh",
    "phase",
    "status",
    "model_scale",
    "candidate_policy",
    "execution",
    "routes",
    "formal_ranking_requires",
    "datasets",
    "recipe",
    "adjustable",
    "validation",
    "transition",
    "lineage",
}
_RECIPE_FIELDS = {
    "source",
    "task_ratio",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "nbs",
    "losses",
    "augmentation",
    "freeze_policy",
    "amp",
    "deterministic",
    "seed",
    "optimizer_steps",
    "validation_interval",
}
_ADJUSTABLE_FIELDS = {
    "batch",
    "fraction",
    "scale",
    "cache",
    "imgsz",
    "device",
    "workers",
    "epochs",
    "patience",
}


def _validate_source_value(path: Path, label: str, value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"source", "value"}:
        raise ConfigContractError(f"{path}: {label} 必須使用 source/value 明示來源")
    if value["source"] not in {"handoff", "local", "derived"}:
        raise ConfigContractError(f"{path}: {label}.source 不支援")
    if value["source"] == "handoff" and value["value"] is not None:
        raise ConfigContractError(f"{path}: {label} 尚未 handoff，不得預填值")


def load_training_template(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    payload = load_yaml(resolved)
    validate_spec_metadata(payload, resolved, spec_path=root / "EXPERIMENT_SPEC.md")
    if set(payload) != _TRAINING_KEYS:
        raise ConfigContractError(
            f"{resolved}: training template 欄位不完整，"
            f"missing={sorted(_TRAINING_KEYS - set(payload))} "
            f"unknown={sorted(set(payload) - _TRAINING_KEYS)}"
        )
    if (
        payload["schema_version"] != 3
        or payload["kind"] != "training_template"
        or payload["model_scale"] != "m"
        or payload["config_id"] not in {"cpu-smoke", "float-main", "float-extension", "quant-qat"}
    ):
        raise ConfigContractError(f"{resolved}: training template metadata 漂移")
    policy = payload["candidate_policy"]
    if (
        not isinstance(policy, dict)
        or policy.get("base_candidates") != ["C0", "C1", "C2", "C3"]
        or policy.get("resolved_from_handoff") is not True
        or policy.get("same_recipe_for_all") is not True
    ):
        raise ConfigContractError(f"{resolved}: 候選公平性 policy 漂移")
    routes = payload["routes"]
    if not isinstance(routes, dict) or set(routes) != {"detect", "pose"}:
        raise ConfigContractError(f"{resolved}: routes 必須恰好是 detect/pose")
    datasets = payload["datasets"]
    expected_datasets = {
        "detect": "configs/data/coco2017.yaml",
        "pose": "configs/data/bbat5-pose.yaml",
        "diagnostic_detect": "configs/data/bbat5-detect.yaml",
    }
    if datasets != expected_datasets:
        raise ConfigContractError(f"{resolved}: 主任務與診斷 dataset 契約漂移")
    recipe = payload["recipe"]
    adjustable = payload["adjustable"]
    if not isinstance(recipe, dict) or set(recipe) != _RECIPE_FIELDS:
        raise ConfigContractError(f"{resolved}: recipe fields 不完整")
    if not isinstance(adjustable, dict) or set(adjustable) != _ADJUSTABLE_FIELDS:
        raise ConfigContractError(
            f"{resolved}: 必須顯式提供 batch/fraction/scale/cache/imgsz/device/workers/epochs/patience"
        )
    for field in _RECIPE_FIELDS - {"source"}:
        _validate_source_value(resolved, f"recipe.{field}", recipe[field])
    for field in _ADJUSTABLE_FIELDS:
        _validate_source_value(resolved, f"adjustable.{field}", adjustable[field])
    if payload["config_id"] == "float-main" and (
            payload["formal_ranking_requires"] != ["detect", "pose"]
            or routes["pose"].get("enabled_by_default") is not False
            or payload["execution"].get("requires_gpu_authorization") is not True
            or payload["execution"].get("pose_opt_in_required") is not True
            or payload["validation"].get("selection_backend") != "float_model"
    ):
        raise ConfigContractError(f"{resolved}: Float 正式排名與 Pose opt-in policy 漂移")
    if payload["config_id"] == "cpu-smoke" and (
            payload["execution"].get("device") != "cpu"
            or payload["execution"].get("force_cuda_hidden") is not True
            or payload["execution"].get("formal_training") is not False
    ):
        raise ConfigContractError(f"{resolved}: CPU smoke 不得使用 CUDA 或宣稱正式訓練")
    if payload["config_id"] == "float-extension":
        expected_transition = {
            "mode": "resume",
            "input": "own_unstripped_continuation_checkpoint",
            "optimizer_state": "restore",
            "gate": "handoff_late_improvement",
            "requires_unstripped_state": True,
            "required_checkpoint_state": [
                "epoch",
                "best_fitness",
                "optimizer",
                "scaler",
                "ema",
                "updates",
            ],
            "required_resume_evidence": [
                "scheduler_position",
                "optimizer_parameter_groups",
            ],
            "stripped_last_or_best_policy": "reject",
        }
        if payload["transition"] != expected_transition:
            raise ConfigContractError(
                f"{resolved}: extension requires unstripped continuation state"
            )
        if payload["validation"].get("selection_backend") != "float_model":
            raise ConfigContractError(f"{resolved}: Float extension 必須以 Float model 驗證")
    if payload["config_id"] == "quant-qat" and (
            payload["validation"].get("selection_backend") != "w8a8_simulation"
            or payload["validation"].get("simulation_only") is not True
    ):
        raise ConfigContractError(f"{resolved}: QAT 只能宣稱 W8A8 simulation")
    return payload


def validate_runtime_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(overrides or {})
    forbidden = set(values) - RUNTIME_OVERRIDE_ALLOWLIST
    if forbidden:
        raise ConfigContractError(f"CLI learning-field overrides are forbidden: {sorted(forbidden)}")
    return values


@dataclass(frozen=True)
class ConfigCheckReport:
    valid: bool
    spec_version: str
    spec_sha256: str
    ultralytics_version: str
    catalog_file: str
    candidate_ids: tuple[str, ...]
    checked_training_files: tuple[str, ...]
    accepted_but_inactive: tuple[str, ...]
    deprecated: tuple[str, ...]
    overridden: tuple[str, ...]
    blocked: tuple[str, ...]
    runtime_overridable: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "spec_version": self.spec_version,
            "spec_sha256": self.spec_sha256,
            "ultralytics_version": self.ultralytics_version,
            "catalog_file": self.catalog_file,
            "candidate_ids": list(self.candidate_ids),
            "checked_training_files": list(self.checked_training_files),
            "accepted_but_inactive": list(self.accepted_but_inactive),
            "deprecated": list(self.deprecated),
            "overridden": list(self.overridden),
            "blocked": list(self.blocked),
            "runtime_overridable": list(self.runtime_overridable),
        }


_EXPECTED_CATALOG = {
    "architectures": {
        "C0": "configs/candidates/c0.yaml",
        "C1": "configs/candidates/c1-e0375.yaml",
        "C2": "configs/candidates/c2-n1.yaml",
        "C3": "configs/candidates/c3-mixed.yaml",
    },
    "training": {
        "cpu-smoke": "configs/training/cpu-smoke.yaml",
        "float-extension": "configs/training/float-extension.yaml",
        "float-main": "configs/training/float-main.yaml",
        "quant-qat": "configs/training/quant-qat.yaml",
    },
    "datasets": {
        "coco2017": "configs/data/coco2017.yaml",
        "bbat5-pose": "configs/data/bbat5-pose.yaml",
        "bbat5-detect": "configs/data/bbat5-detect.yaml",
        "bbat5-pose-search": "configs/data/bbat5-pose-search.yaml",
        "bbat5-detect-search": "configs/data/bbat5-detect-search.yaml",
    },
    "quantization": "configs/quant/w8a8-simulation.yaml",
    "schemas": {
        "catalog": "configs/schema/catalog.schema.yaml",
        "candidate": "configs/schema/candidate.schema.yaml",
        "training": "configs/schema/training.schema.yaml",
        "dataset": "configs/schema/dataset.schema.yaml",
        "quantization": "configs/schema/quantization.schema.yaml",
        "handoff": "configs/schema/handoff.schema.yaml",
    },
}


def _validate_catalog(root: Path) -> dict[str, Any]:
    path = root / "configs/catalog.yaml"
    payload = load_yaml(path)
    validate_spec_metadata(payload, path, spec_path=root / "EXPERIMENT_SPEC.md")
    fixed = {
        "schema_version": 3,
        "kind": "config_catalog",
        "readme": "configs/README.md",
        "authoritative_spec": "EXPERIMENT_SPEC.md",
        "handoff_example": "handoff-manifest.example.json",
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise ConfigContractError(f"{path}: catalog 固定欄位漂移")
    for section, expected in _EXPECTED_CATALOG.items():
        if payload.get(section) != expected:
            raise ConfigContractError(f"{path}: catalog.{section} 漂移")
    referenced: list[str] = [
        *payload["architectures"].values(),
        *payload["training"].values(),
        *payload["datasets"].values(),
        payload["quantization"],
        *payload["schemas"].values(),
        payload["handoff_example"],
        payload["readme"],
        payload["authoritative_spec"],
    ]
    missing = [relative for relative in referenced if not (root / relative).is_file()]
    if missing:
        raise ConfigContractError(f"{path}: catalog 引用不存在檔案 {missing}")
    return payload


def _normalize_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    raise ConfigContractError("dataset names 必須是 list 或 mapping")


def _validate_datasets(root: Path, catalog: Mapping[str, Any]) -> None:
    datasets = {
        name: load_yaml(root / relative) for name, relative in catalog["datasets"].items()
    }
    for name, payload in datasets.items():
        path = root / catalog["datasets"][name]
        validate_spec_metadata(payload, path, spec_path=root / "EXPERIMENT_SPEC.md")
        if payload.get("schema_version") != 3 or payload.get("kind") != "dataset":
            raise ConfigContractError(f"{path}: dataset schema/kind 漂移")
    coco = datasets["coco2017"]
    coco_names = _normalize_names(coco.get("names"))
    if len(coco_names) != 80 or coco_names.get(0) != "person":
        raise ConfigContractError("COCO 主線必須保留 80 classes，class 0 必須是 person")
    if coco.get("project_metadata", {}).get("role") != "main_detect":
        raise ConfigContractError("COCO dataset role 必須是 main_detect")
    pose = datasets["bbat5-pose"]
    detect = datasets["bbat5-detect"]
    pose_search = datasets["bbat5-pose-search"]
    detect_search = datasets["bbat5-detect-search"]
    expected_names = {0: "ball", 1: "bat"}
    if (
        _normalize_names(pose.get("names")) != expected_names
        or pose.get("kpt_shape") != [2, 3]
        or pose.get("test") is not None
        or pose.get("project_metadata", {}).get("role") != "main_pose"
    ):
        raise ConfigContractError("BBAT5 Pose 主任務契約漂移")
    if (
        _normalize_names(detect.get("names")) != expected_names
        or detect.get("test") is not None
        or detect.get("project_metadata", {}).get("role") != "diagnostic_detect"
    ):
        raise ConfigContractError("BBAT5 Detect 必須維持 2-class 診斷角色")
    if (
        _normalize_names(pose_search.get("names")) != expected_names
        or pose_search.get("kpt_shape") != [2, 3]
        or pose_search.get("test") is not None
        or pose_search.get("project_metadata", {}).get("role") != "search_pose"
        or pose_search.get("project_metadata", {}).get("source_scope")
        != "formal_train_only"
        or pose_search.get("project_metadata", {}).get("formal_val_excluded") is not True
        or Path(str(pose_search.get("train"))).name != "search-train.txt"
        or Path(str(pose_search.get("val"))).name != "search-val.txt"
    ):
        raise ConfigContractError("BBAT5 Pose search 必須只使用 formal train 子分割")
    if (
        _normalize_names(detect_search.get("names")) != expected_names
        or detect_search.get("test") is not None
        or detect_search.get("project_metadata", {}).get("role")
        != "search_diagnostic_detect"
        or detect_search.get("project_metadata", {}).get("source_scope")
        != "formal_train_only"
        or detect_search.get("project_metadata", {}).get("formal_val_excluded") is not True
        or Path(str(detect_search.get("train"))).name != "search-train.txt"
        or Path(str(detect_search.get("val"))).name != "search-val.txt"
    ):
        raise ConfigContractError("BBAT5 Detect search 必須維持 formal-train-only 診斷角色")
    grouped = (pose, detect, pose_search, detect_search)
    if len({item.get("group_key") for item in grouped}) != 1 or len(
        {item.get("split_seed") for item in grouped}
    ) != 1:
        raise ConfigContractError("BBAT5 Detect/Pose 必須共用 grouped assignment")


def _validate_quantization(root: Path, relative: str) -> None:
    path = root / relative
    payload = load_yaml(path)
    validate_spec_metadata(payload, path, spec_path=root / "EXPERIMENT_SPEC.md")
    expected = {
        "C0": True,
        "C1": "pending_user_decision",
        "C2": "pending_user_decision",
        "C3": "pending_user_decision",
    }
    if (
        payload.get("schema_version") != 3
        or payload.get("kind") != "quantization"
        or payload.get("simulation_only") is not True
        or payload.get("eligibility") != expected
        or set(payload.get("stages", {})) != {"Q0", "Q1", "Q2"}
    ):
        raise ConfigContractError(f"{path}: 逐候選量化契約漂移")


def _validate_schema_metadata(root: Path, schemas: Mapping[str, str]) -> None:
    expected_sha = file_sha256(root / "EXPERIMENT_SPEC.md")
    for relative in schemas.values():
        path = root / relative
        payload = load_yaml(path)
        if (
            payload.get("x-spec-version") != SPEC_VERSION
            or payload.get("x-spec-sha256") != expected_sha
            or payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        ):
            raise ConfigContractError(f"{path}: schema 規格 metadata 漂移")


def check_configs(project_root: str | Path = PROJECT_ROOT) -> ConfigCheckReport:
    root = Path(project_root).resolve()
    import ultralytics

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise ConfigContractError(
            f"Ultralytics {ultralytics.__version__} 不符合固定版本 {EXPECTED_ULTRALYTICS_VERSION}"
        )
    catalog = _validate_catalog(root)
    candidates = load_candidate_specs(
        root / "configs/candidates", spec_path=root / "EXPERIMENT_SPEC.md"
    )
    checked_training: list[str] = []
    for relative in sorted(catalog["training"].values()):
        load_training_template(relative, project_root=root)
        checked_training.append(relative)
    _validate_datasets(root, catalog)
    _validate_quantization(root, catalog["quantization"])
    _validate_schema_metadata(root, catalog["schemas"])

    forbidden_formal = [
        root / "configs/fusion/source-pair.template.yaml",
        root / "configs/candidates/c3-p5-only.yaml",
        root / "configs/candidates/r1-rep.yaml",
    ]
    if any(path.exists() for path in forbidden_formal):
        raise ConfigContractError("v2 不得保留重複融合範本或未獲使用者核准的條件候選")
    return ConfigCheckReport(
        valid=True,
        spec_version=SPEC_VERSION,
        spec_sha256=file_sha256(root / "EXPERIMENT_SPEC.md"),
        ultralytics_version=ultralytics.__version__,
        catalog_file="configs/catalog.yaml",
        candidate_ids=tuple(candidates),
        checked_training_files=tuple(checked_training),
        accepted_but_inactive=(
            "Pose 正式執行：等待使用者 opt-in",
            "Q2 QAT：等待 Float 結果、量化資格與 GPU 授權",
        ),
        deprecated=(
            "architecture_1 direct handoff：改由 yolo_combine winner handoff",
            "batch_size：請使用 Ultralytics 正式鍵名 batch",
            "fusion template：融合選型屬於 yolo_combine",
        ),
        overridden=(
            "name/project/device/workers/cache：只允許在 effective config 的 runtime 層覆寫",
        ),
        blocked=(
            "上游 winner training recipe 尚未 handoff",
            "Float extension：等待 late gate 與未 strip continuation state",
            "Pose RLE active evidence：等待 handoff model/loss dry-run",
            "正式 CUDA/GPU 測試與長訓練尚未獲授權",
        ),
        runtime_overridable=tuple(sorted(RUNTIME_OVERRIDE_ALLOWLIST)),
    )


@dataclass(frozen=True)
class EffectiveTrainingConfig:
    """handoff 通過後產生的完整 effective training 設定。"""

    config_id: str
    args: dict[str, Any]
    sources: tuple[Path, ...]
    sha256: str


def resolve_training_template(
    path: str | Path,
    *,
    handoff_recipe: Mapping[str, Any],
    runtime_overrides: Mapping[str, Any] | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> EffectiveTrainingConfig:
    """把 source=handoff 欄位解析成單一完整 mapping；缺值即停止。"""

    root = Path(project_root).resolve()
    template_path = Path(path)
    if not template_path.is_absolute():
        template_path = root / template_path
    payload = load_training_template(template_path, project_root=root)
    values: dict[str, Any] = {}
    for section in ("recipe", "adjustable"):
        for name, declaration in payload[section].items():
            if name == "source":
                continue
            source = declaration["source"]
            value = declaration["value"]
            if source == "handoff":
                if name not in handoff_recipe:
                    raise ConfigContractError(f"handoff training recipe 缺少 {name}")
                value = handoff_recipe[name]
            elif source == "derived":
                if value == "parent_lr0_times_0.1":
                    if "lr0" not in handoff_recipe:
                        raise ConfigContractError("handoff training recipe 缺少 lr0")
                    value = float(handoff_recipe["lr0"]) * 0.1
                else:
                    raise ConfigContractError(f"不支援的 derived training value: {value}")
            values[name] = value
    values.update(validate_runtime_overrides(runtime_overrides))
    unresolved = sorted(name for name, value in values.items() if value is None)
    if unresolved:
        raise ConfigContractError(f"effective training 仍有未解析欄位：{unresolved}")
    return EffectiveTrainingConfig(
        config_id=str(payload["config_id"]),
        args=values,
        sources=(template_path.resolve(),),
        sha256=composite_sha256((template_path.resolve(),)),
    )


def load_formal_training_config(
    path: str | Path,
    *,
    candidate_id: str = "C0",
    project_root: str | Path = PROJECT_ROOT,
) -> EffectiveTrainingConfig:
    """相容名稱；v2 handoff 前不得假裝產生完整正式設定。"""

    del candidate_id
    raise ConfigContractError(
        f"{path}: v2 必須使用 resolve_training_template 並提供 yolo_combine handoff recipe"
    )


def compose_training_config(**_: Any) -> EffectiveTrainingConfig:
    raise ConfigContractError("v2 不再依 task/stage 拼接設定；請使用單一 training YAML")


def manifest_hashes(
    *,
    spec_path: Path,
    architecture_path: Path,
    training: EffectiveTrainingConfig,
    dataset_path: Path,
    parent_checkpoint: Path,
    handoff_manifest: Path | None = None,
) -> dict[str, str]:
    payload = {
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(spec_path),
        "architecture_yaml_sha256": file_sha256(architecture_path),
        "training_yaml_sha256": training.sha256,
        "dataset_yaml_sha256": file_sha256(dataset_path),
        "parent_checkpoint_sha256": file_sha256(parent_checkpoint),
    }
    if handoff_manifest is not None:
        payload["handoff_manifest_sha256"] = file_sha256(handoff_manifest)
    return payload


def write_effective_training(config: EffectiveTrainingConfig, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "config_id": config.config_id,
        "source_yaml_sha256": config.sha256,
        "training": config.args,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
