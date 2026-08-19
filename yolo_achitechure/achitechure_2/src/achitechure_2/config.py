"""Fail-closed executable configuration derived from EXPERIMENT_SPEC.md."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .lite_c3k2 import LiteC3k2Config

SPEC_VERSION = "1.2.0"
EXPECTED_ULTRALYTICS_VERSION = "8.4.90"
TARGET_LAYERS = (6, 8, 13, 19)
P5_ONLY_LAYER = (8,)
PROTECTED_C3K2_LAYERS = (2, 4, 16, 22)
DETECT_INPUTS = (16, 19, 22)
ATTENTION_PATHS = ("model.10.m.0.attn", "model.22.m.0.1.attn")
STRIDES = (8, 16, 32)
RUNTIME_OVERRIDE_ALLOWLIST = frozenset({"model", "name", "project", "device", "workers", "cache"})
FORBIDDEN_COMBINATIONS = ("C1+C2", "C1+C3", "C2+C3")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "EXPERIMENT_SPEC.md"
CANDIDATE_DIR = PROJECT_ROOT / "configs/candidates"
FORMAL_TRAINING_FILES = {
    ("detect", "D0"): "configs/training/detect/d0-smoke.yaml",
    ("detect", "D1"): "configs/training/detect/d1-main.yaml",
    ("detect", "D2"): "configs/training/detect/d2-extension.yaml",
    ("detect", "Q2"): "configs/training/detect/q2-qat.yaml",
    ("pose", "P0"): "configs/training/pose/p0-smoke.yaml",
    ("pose", "P1"): "configs/training/pose/p1-head-only.yaml",
    ("pose", "P2"): "configs/training/pose/p2-neck-head.yaml",
    ("pose", "P3"): "configs/training/pose/p3-full.yaml",
    ("pose", "P4"): "configs/training/pose/p4-extension.yaml",
}
FORMAL_TRAINING_IDS = {
    ("detect", "D0"): "detect-d0-smoke",
    ("detect", "D1"): "detect-d1-main",
    ("detect", "D2"): "detect-d2-extension",
    ("detect", "Q2"): "detect-q2-qat",
    ("pose", "P0"): "pose-p0-smoke",
    ("pose", "P1"): "pose-p1-head-only",
    ("pose", "P2"): "pose-p2-neck-head",
    ("pose", "P3"): "pose-p3-full",
    ("pose", "P4"): "pose-p4-extension",
}
PERMANENT_FROZEN_MODULES = (
    "model.16.p3_masf",
    "model.10.m.0.attn",
    "model.22.m.0.1.attn",
)
FORMAL_STAGE_RULES = {
    ("detect", "D0"): {"epochs": 3, "patience": 3, "mode": "fresh", "freeze": None},
    ("detect", "D1"): {"epochs": 100, "patience": 20, "mode": "fresh", "freeze": None},
    ("detect", "D2"): {"epochs": 140, "patience": 15, "mode": "resume", "freeze": None},
    ("detect", "Q2"): {"epochs": 15, "patience": 5, "mode": "fresh", "freeze": None},
    ("pose", "P0"): {"epochs": 3, "patience": 3, "mode": "fresh", "freeze": None},
    ("pose", "P1"): {"epochs": 10, "patience": 5, "mode": "fresh", "freeze": list(range(23))},
    ("pose", "P2"): {"epochs": 20, "patience": 8, "mode": "fresh", "freeze": 11},
    ("pose", "P3"): {"epochs": 100, "patience": 20, "mode": "fresh", "freeze": None},
    ("pose", "P4"): {"epochs": 130, "patience": 10, "mode": "resume", "freeze": None},
}


class ConfigContractError(ValueError):
    """A configuration cannot be proven to match the authoritative spec."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(path: str | Path) -> str:
    return file_sha256(path)


def composite_sha256(paths: Iterable[Path]) -> str:
    """Hash an ordered set of YAML files including their names and boundaries."""

    digest = hashlib.sha256()
    for path in paths:
        resolved = Path(path).resolve()
        digest.update(resolved.name.encode())
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 應為 YAML mapping")
    return payload


def validate_spec_metadata(
    payload: dict[str, Any],
    path: str | Path,
    *,
    spec_path: str | Path = SPEC_PATH,
) -> None:
    expected_sha = file_sha256(spec_path)
    if payload.get("spec_version") != SPEC_VERSION:
        raise ConfigContractError(f"{path}: spec_version {payload.get('spec_version')!r} != {SPEC_VERSION!r}")
    if payload.get("spec_sha256") != expected_sha:
        raise ConfigContractError(f"{path}: spec_sha256 與 EXPERIMENT_SPEC.md 不符")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    target_layers: tuple[int, ...]
    lite: LiteC3k2Config | None
    declared_factors: tuple[float, int, str, bool]
    config_path: Path
    base_candidate: str | None = None
    trigger: str | None = None


_EXPECTED_CANDIDATES: dict[str, dict[str, Any]] = {
    "C0": {"targets": (), "factors": (0.5, 2, "3x3_3x3", False), "base": None, "trigger": None},
    "C1": {"targets": TARGET_LAYERS, "factors": (0.375, 2, "3x3_3x3", False), "base": None, "trigger": None},
    "C2": {"targets": TARGET_LAYERS, "factors": (0.5, 1, "3x3_3x3", False), "base": None, "trigger": None},
    "C3": {"targets": TARGET_LAYERS, "factors": (0.5, 2, "1x1_3x3", False), "base": None, "trigger": None},
    "C3-P5": {
        "targets": P5_ONLY_LAYER,
        "factors": (0.5, 2, "1x1_3x3", False),
        "base": "C3",
        "trigger": "C3 drop > 0.008",
    },
    "R1": {
        "targets": TARGET_LAYERS,
        "factors": (0.5, 1, "3x3_3x3", True),
        "base": "C2",
        "trigger": "C2 is CONDITIONAL and latency or GFLOPs gain >= 0.08",
    },
}


def _candidate_from_yaml(path: Path, *, spec_path: Path) -> CandidateSpec:
    payload = load_yaml(path)
    validate_spec_metadata(payload, path, spec_path=spec_path)
    required = {
        "schema_version",
        "kind",
        "title_zh",
        "summary_zh",
        "status",
        "task_contracts",
        "scale",
        "candidate_id",
        "parent_policy",
        "target_layers",
        "protected_layers",
        "factors",
        "inherited",
        "heads",
        "conditions",
        "forbidden_combinations",
        "standalone_loadable",
        "builder",
        "architecture_delta",
    }
    unknown = set(payload) - required - {"spec_version", "spec_sha256"}
    missing = required - set(payload)
    if unknown or missing:
        raise ConfigContractError(f"{path}: missing={sorted(missing)} unknown={sorted(unknown)}")
    if (
        payload["schema_version"] != 2
        or payload["kind"] != "architecture_candidate"
        or payload["task_contracts"] != ["detect", "pose"]
        or payload["scale"] != "m"
        or payload["protected_layers"] != list(PROTECTED_C3K2_LAYERS)
        or tuple(payload["forbidden_combinations"]) != FORBIDDEN_COMBINATIONS
        or payload["standalone_loadable"] is not False
        or payload["builder"] != "achitechure_2"
    ):
        raise ConfigContractError(f"{path}: invariant architecture metadata drift")
    if payload["parent_policy"] != {
        "source": "accepted_float_pwl",
        "independence": "same_parent",
        "seed": 0,
    }:
        raise ConfigContractError(f"{path}: parent policy drift")
    if payload["inherited"] != {
        "masf": {
            "variant": "architecture_1_selected",
            "frozen": True,
            "path": "model.16.p3_masf",
        },
        "attention": {
            "frozen": True,
            "paths": list(ATTENTION_PATHS),
        },
    }:
        raise ConfigContractError(f"{path}: inherited module contract drift")
    expected_heads = {
        task: {
            "type": "Detect" if task == "detect" else "Pose26",
            "inputs": list(DETECT_INPUTS),
            "strides": list(STRIDES),
        }
        for task in ("detect", "pose")
    }
    if payload["heads"] != expected_heads:
        raise ConfigContractError(f"{path}: task head contract drift")

    factors = payload["factors"]
    if set(factors) != {"e", "inner_n", "kernel_mode", "use_rep"}:
        raise ConfigContractError(f"{path}: factors must be exactly e/inner_n/kernel_mode/use_rep")
    if (
        not isinstance(factors["e"], (int, float))
        or isinstance(factors["e"], bool)
        or not isinstance(factors["inner_n"], int)
        or isinstance(factors["inner_n"], bool)
        or factors["kernel_mode"] not in {"3x3_3x3", "1x1_3x3"}
        or not isinstance(factors["use_rep"], bool)
    ):
        raise ConfigContractError(f"{path}: invalid factor type/value")
    declared_factors = (
        float(factors["e"]),
        int(factors["inner_n"]),
        str(factors["kernel_mode"]),
        factors["use_rep"],
    )
    candidate_id = str(payload["candidate_id"])
    conditions = payload["conditions"]
    if set(conditions) != {"base_candidate", "trigger"}:
        raise ConfigContractError(f"{path}: invalid conditions")
    lite = None
    if candidate_id != "C0":
        lite = LiteC3k2Config(
            e=float(factors["e"]),
            inner_n=int(factors["inner_n"]),
            kernel_mode=str(factors["kernel_mode"]),
            use_rep=bool(factors["use_rep"]),
        )
    return CandidateSpec(
        candidate_id,
        tuple(int(value) for value in payload["target_layers"]),
        lite,
        declared_factors,
        path.resolve(),
        conditions["base_candidate"],
        conditions["trigger"],
    )


def load_candidate_specs(
    config_dir: str | Path = CANDIDATE_DIR,
    *,
    spec_path: str | Path = SPEC_PATH,
) -> dict[str, CandidateSpec]:
    paths = sorted(Path(config_dir).glob("*.yaml"))
    specs = [_candidate_from_yaml(path, spec_path=Path(spec_path)) for path in paths]
    candidates = {item.candidate_id: item for item in specs}
    if len(candidates) != len(specs):
        raise ConfigContractError("duplicate candidate_id")
    if set(candidates) != set(_EXPECTED_CANDIDATES):
        raise ConfigContractError(
            f"candidate matrix drift: got {sorted(candidates)}, expected {sorted(_EXPECTED_CANDIDATES)}"
        )
    for candidate_id, expected in _EXPECTED_CANDIDATES.items():
        item = candidates[candidate_id]
        if (
            item.target_layers != expected["targets"]
            or item.declared_factors != expected["factors"]
            or item.base_candidate != expected["base"]
            or item.trigger != expected["trigger"]
        ):
            raise ConfigContractError(f"{item.config_path}: {candidate_id} matrix entry drift")
    return candidates


CANDIDATES = load_candidate_specs()


@dataclass(frozen=True)
class EffectiveTrainingConfig:
    task: str
    candidate_id: str
    stage: str
    args: dict[str, Any]
    sources: tuple[Path, ...]
    sha256: str
    transition_mode: str
    transition_input: str
    execution_policy: str
    config_id: str
    title_zh: str

    @property
    def dataset_path(self) -> Path:
        value = Path(str(self.args["data"]))
        return value if value.is_absolute() else PROJECT_ROOT / value


def formal_training_path(root: Path, task: str, stage: str) -> Path:
    """Return the one authoritative training YAML for a task/stage pair."""

    key = (task.lower(), stage.upper())
    try:
        relative = FORMAL_TRAINING_FILES[key]
    except KeyError as error:
        raise ConfigContractError(f"沒有正式訓練設定：task={key[0]!r}, stage={key[1]!r}") from error
    return root / relative


def load_formal_training_config(
    path: str | Path,
    *,
    candidate_id: str,
    project_root: str | Path = PROJECT_ROOT,
) -> EffectiveTrainingConfig:
    """Load one complete formal training YAML without implicit composition."""

    root = Path(project_root).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    payload = load_yaml(config_path)
    validate_spec_metadata(payload, config_path, spec_path=root / "EXPERIMENT_SPEC.md")
    required = {
        "schema_version",
        "spec_version",
        "spec_sha256",
        "kind",
        "config_id",
        "title_zh",
        "summary_zh",
        "task",
        "stage",
        "status",
        "candidate_policy",
        "execution",
        "dataset",
        "transition",
        "ranking",
        "frozen_modules",
        "derived_training",
        "training",
    }
    if set(payload) != required or payload.get("schema_version") != 2:
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        raise ConfigContractError(f"{config_path}: 正式訓練欄位不完整，missing={missing} unknown={unknown}")
    if payload.get("kind") != "formal_training":
        raise ConfigContractError(f"{config_path}: kind 必須是 formal_training")
    task = str(payload["task"]).lower()
    stage = str(payload["stage"]).upper()
    key = (task, stage)
    if payload.get("config_id") != FORMAL_TRAINING_IDS.get(key):
        raise ConfigContractError(f"{config_path}: config_id、task、stage 不一致")
    if not isinstance(payload.get("title_zh"), str) or not isinstance(payload.get("summary_zh"), str):
        raise ConfigContractError(f"{config_path}: 必須提供 title_zh 與 summary_zh")
    execution = payload.get("execution")
    expected_execution = {
        "launcher": "achitechure_2",
        "requires_execute_flag": True,
        "pose_opt_in_required": task == "pose",
        "enabled_by_default": payload.get("status") in {"required_smoke", "required_main"},
        "ultralytics_yaml_direct": False,
        "runtime_overridable": ["model", "name", "project", "device", "workers", "cache"],
    }
    if execution != expected_execution:
        raise ConfigContractError(f"{config_path}: execution policy 漂移")
    candidate_policy = payload.get("candidate_policy")
    expected_policy_mode = (
        "c0_or_recorded_c_best" if task == "pose" or stage == "Q2" else "architecture_matrix"
    )
    if not isinstance(candidate_policy, dict) or candidate_policy.get("mode") != expected_policy_mode:
        raise ConfigContractError(f"{config_path}: candidate policy 漂移")
    if candidate_id not in CANDIDATES and candidate_id != "C_best":
        raise ConfigContractError(f"{config_path}: 未知 candidate_id {candidate_id!r}")
    if candidate_id == "C_best" and expected_policy_mode != "c0_or_recorded_c_best":
        raise ConfigContractError(f"{config_path}: C_best 只適用於下游正式設定")
    transition = payload.get("transition")
    if not isinstance(transition, dict) or set(transition) != {
        "mode",
        "checkpoint_role",
        "gate",
        "description_zh",
    }:
        raise ConfigContractError(f"{config_path}: transition 契約不完整")
    if transition["mode"] not in {"fresh", "resume"}:
        raise ConfigContractError(f"{config_path}: transition.mode 不支援")
    ranking = payload.get("ranking")
    if not isinstance(ranking, dict) or set(ranking) != {
        "architecture_ranking",
        "research_metric",
        "selection_backend",
    }:
        raise ConfigContractError(f"{config_path}: ranking 契約不完整")
    if ranking["selection_backend"] != "bit_true_pwl":
        raise ConfigContractError(f"{config_path}: selection backend 必須是 bit_true_pwl")
    if payload.get("frozen_modules") != list(PERMANENT_FROZEN_MODULES):
        raise ConfigContractError(f"{config_path}: 永久凍結模組漂移")
    if not isinstance(payload.get("derived_training"), dict):
        raise ConfigContractError(f"{config_path}: derived_training 必須是 mapping")
    args = payload.get("training")
    if not isinstance(args, dict):
        raise ConfigContractError(f"{config_path}: training 必須是完整 mapping")
    if args.get("task") != task or args.get("data") != payload.get("dataset"):
        raise ConfigContractError(f"{config_path}: task 或 dataset 與 training 內容不一致")
    expected_stage = FORMAL_STAGE_RULES[key]
    for field in ("epochs", "patience", "freeze"):
        if args.get(field) != expected_stage[field]:
            raise ConfigContractError(
                f"{config_path}: {field}={args.get(field)!r}，預期 {expected_stage[field]!r}"
            )
    if transition["mode"] != expected_stage["mode"]:
        raise ConfigContractError(f"{config_path}: transition.mode 與 stage 規則不一致")
    expected_dataset = (
        "configs/data/pose-grouped.yaml" if task == "pose" else "configs/data/coco2017.yaml"
    )
    if payload.get("dataset") != expected_dataset:
        raise ConfigContractError(f"{config_path}: dataset 應為 {expected_dataset}")
    expected_metric = "metrics/mAP50-95(P)" if task == "pose" else "metrics/mAP50-95(B)"
    if ranking["research_metric"] != expected_metric:
        raise ConfigContractError(f"{config_path}: research metric 漂移")
    derived = payload["derived_training"]
    if stage == "Q2":
        expected_derived = {
            "lr0": {
                "formula": "parent_checkpoint_lr0 * 0.1",
                "expected_if_parent_uses_spec": 0.000038,
            }
        }
        if derived != expected_derived or args.get("lr0") != 0.000038:
            raise ConfigContractError(f"{config_path}: Q2 lr0 衍生契約漂移")
    elif stage in {"D2", "P4"}:
        expected_derived = {
            "resume": {
                "formula": "checkpoint_argument_path",
                "required_filename": "last.pt",
            }
        }
        if derived != expected_derived or args.get("resume") is not True:
            raise ConfigContractError(f"{config_path}: resume path 衍生契約漂移")
    elif derived:
        raise ConfigContractError(f"{config_path}: 本 stage 不得宣告 derived_training")
    args = dict(args)
    args["name"] = f"{task}-{candidate_id.lower().replace('_', '-')}-{stage.lower()}"
    return EffectiveTrainingConfig(
        task,
        candidate_id,
        stage,
        args,
        (config_path,),
        file_sha256(config_path),
        str(transition["mode"]),
        str(transition["checkpoint_role"]),
        "user_opt_in" if task == "pose" else "required",
        str(payload["config_id"]),
        str(payload["title_zh"]),
    )


def compose_training_config(
    *,
    project_root: str | Path = PROJECT_ROOT,
    task: str,
    candidate_id: str,
    stage: str,
) -> EffectiveTrainingConfig:
    """Compatibility interface resolving task/stage to the same formal YAML."""

    root = Path(project_root).resolve()
    path = formal_training_path(root, task, stage)
    return load_formal_training_config(path, candidate_id=candidate_id, project_root=root)


def validate_runtime_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    overrides = dict(overrides or {})
    forbidden = set(overrides) - RUNTIME_OVERRIDE_ALLOWLIST
    if forbidden:
        raise ConfigContractError(f"CLI learning-field overrides are forbidden: {sorted(forbidden)}")
    return overrides


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
    runtime_overridable: tuple[str, ...]
    optional_routes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "spec_version": self.spec_version,
            "spec_sha256": self.spec_sha256,
            "ultralytics_version": self.ultralytics_version,
            "catalog_file": self.catalog_file,
            "candidate_ids": self.candidate_ids,
            "checked_training_files": self.checked_training_files,
            "accepted_but_inactive": self.accepted_but_inactive,
            "runtime_overridable": self.runtime_overridable,
            "optional_routes": self.optional_routes,
        }


def _validate_catalog(root: Path) -> dict[str, Any]:
    path = root / "configs/catalog.yaml"
    catalog = load_yaml(path)
    validate_spec_metadata(catalog, path, spec_path=root / "EXPERIMENT_SPEC.md")
    required = {
        "schema_version",
        "spec_version",
        "spec_sha256",
        "kind",
        "title_zh",
        "readme",
        "authoritative_spec",
        "architectures",
        "training",
        "datasets",
        "quantization",
        "fusion_template",
        "schemas",
    }
    if set(catalog) != required or catalog.get("schema_version") != 2:
        raise ConfigContractError(f"{path}: catalog 欄位不完整")
    if catalog.get("kind") != "config_catalog":
        raise ConfigContractError(f"{path}: kind 必須是 config_catalog")
    expected_architectures = {
        "C0": "configs/candidates/c0.yaml",
        "C1": "configs/candidates/c1-e0375.yaml",
        "C2": "configs/candidates/c2-n1.yaml",
        "C3": "configs/candidates/c3-mixed.yaml",
        "C3-P5": "configs/candidates/c3-p5-only.yaml",
        "R1": "configs/candidates/r1-rep.yaml",
    }
    expected_training = {value: FORMAL_TRAINING_FILES[key] for key, value in FORMAL_TRAINING_IDS.items()}
    expected_datasets = {
        "coco2017": "configs/data/coco2017.yaml",
        "pose-grouped": "configs/data/pose-grouped.yaml",
    }
    expected_schemas = {
        "catalog": "configs/schema/catalog.schema.yaml",
        "candidate": "configs/schema/candidate.schema.yaml",
        "training": "configs/schema/training.schema.yaml",
        "dataset": "configs/schema/dataset.schema.yaml",
        "quantization": "configs/schema/quantization.schema.yaml",
        "fusion": "configs/schema/fusion.schema.yaml",
    }
    if catalog.get("architectures") != expected_architectures:
        raise ConfigContractError(f"{path}: architecture 目錄與正式矩陣不一致")
    if catalog.get("training") != expected_training:
        raise ConfigContractError(f"{path}: training 目錄與正式矩陣不一致")
    if catalog.get("datasets") != expected_datasets:
        raise ConfigContractError(f"{path}: dataset 目錄不一致")
    if catalog.get("schemas") != expected_schemas:
        raise ConfigContractError(f"{path}: schema 目錄不一致")
    if catalog.get("quantization") != "configs/quant/w8a8-simulation.yaml":
        raise ConfigContractError(f"{path}: quantization 路徑不一致")
    if catalog.get("fusion_template") != "configs/fusion/source-pair.template.yaml":
        raise ConfigContractError(f"{path}: fusion template 路徑不一致")
    referenced: list[str] = []
    for field in ("architectures", "training", "datasets", "schemas"):
        value = catalog.get(field)
        if not isinstance(value, dict):
            raise ConfigContractError(f"{path}: {field} 必須是 mapping")
        referenced.extend(str(item) for item in value.values())
    referenced.extend([str(catalog["quantization"]), str(catalog["fusion_template"])])
    missing = sorted(item for item in referenced if not (root / item).is_file())
    if missing:
        raise ConfigContractError(f"{path}: catalog 引用不存在的檔案 {missing}")
    actual_training = {
        str(item.relative_to(root))
        for item in (root / "configs/training").glob("**/*.yaml")
    }
    if actual_training != set(expected_training.values()):
        raise ConfigContractError(
            f"{path}: 未登錄或缺少 training YAML，actual={sorted(actual_training)}"
        )
    for schema_path in catalog["schemas"].values():
        schema = load_yaml(root / schema_path)
        if (
            schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
            or schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
        ):
            raise ConfigContractError(f"{root / schema_path}: 不是正式 Draft 2020-12 schema")
    return catalog


def check_configs(project_root: str | Path = PROJECT_ROOT) -> ConfigCheckReport:
    """Validate the catalog and every formal YAML against the authoritative spec."""

    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg

    root = Path(project_root).resolve()
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise ConfigContractError(
            f"Ultralytics {ultralytics.__version__} != required {EXPECTED_ULTRALYTICS_VERSION}"
        )
    catalog = _validate_catalog(root)
    candidates = load_candidate_specs(root / "configs/candidates", spec_path=root / "EXPERIMENT_SPEC.md")

    checked: list[str] = []
    for config_id, relative in catalog["training"].items():
        candidate_id = "C_best" if config_id.startswith("pose-") else "C0"
        config = load_formal_training_config(
            root / relative,
            candidate_id=candidate_id,
            project_root=root,
        )
        forwarded = {key: value for key, value in config.args.items() if key != "name"}
        unknown = set(forwarded) - set(DEFAULT_CFG_DICT)
        if unknown:
            raise ConfigContractError(f"{root / relative}: Ultralytics 不支援 {sorted(unknown)}")
        try:
            get_cfg(overrides=forwarded)
        except (TypeError, ValueError) as error:
            raise ConfigContractError(f"{root / relative}: Ultralytics 設定值無效：{error}") from error
        checked.append(relative)

    detect_formal = {
        stage: compose_training_config(
            project_root=root, task="detect", candidate_id="C0", stage=stage
        ).args
        for stage in ("D0", "D1", "D2", "Q2")
    }
    pose_formal = {
        stage: compose_training_config(
            project_root=root, task="pose", candidate_id="C0", stage=stage
        ).args
        for stage in ("P0", "P1", "P2", "P3", "P4")
    }
    stage_fields = {"epochs", "patience", "freeze", "resume", "name"}
    detect_reference = {
        key: value for key, value in detect_formal["D1"].items() if key not in stage_fields
    }
    for stage in ("D0", "D2"):
        comparable = {
            key: value for key, value in detect_formal[stage].items() if key not in stage_fields
        }
        if comparable != detect_reference:
            raise ConfigContractError(f"Detect {stage} 與 D1 的共同訓練欄位漂移")
    q2_comparable = {
        key: value
        for key, value in detect_formal["Q2"].items()
        if key not in stage_fields | {"lr0"}
    }
    d1_without_lr = {key: value for key, value in detect_reference.items() if key != "lr0"}
    if q2_comparable != d1_without_lr:
        raise ConfigContractError("Detect Q2 除 lr0 與 stage budget 外不得漂移")
    pose_reference = {
        key: value for key, value in pose_formal["P3"].items() if key not in stage_fields
    }
    for stage, args in pose_formal.items():
        comparable = {key: value for key, value in args.items() if key not in stage_fields}
        if comparable != pose_reference:
            raise ConfigContractError(f"Pose {stage} 與 P3 的共同訓練欄位漂移")

    for candidate_id in CANDIDATES:
        for stage in ("D0", "D1", "D2"):
            compose_training_config(
                project_root=root,
                task="detect",
                candidate_id=candidate_id,
                stage=stage,
            )
    for candidate_id in ("C0", "C_best"):
        compose_training_config(
            project_root=root,
            task="detect",
            candidate_id=candidate_id,
            stage="Q2",
        )
        for stage in ("P0", "P1", "P2", "P3", "P4"):
            compose_training_config(
                project_root=root,
                task="pose",
                candidate_id=candidate_id,
                stage=stage,
            )

    quant_path = root / str(catalog["quantization"])
    quant = load_yaml(quant_path)
    validate_spec_metadata(quant, quant_path, spec_path=root / "EXPERIMENT_SPEC.md")
    quant_fields = {
        "schema_version",
        "spec_version",
        "spec_sha256",
        "kind",
        "title_zh",
        "summary_zh",
        "status",
        "simulation_only",
        "targets",
        "weight",
        "activation",
        "include_masf_standard_conv",
        "exclude_custom_roots",
        "qat",
    }
    if (
        set(quant) != quant_fields
        or quant.get("schema_version") != 2
        or quant.get("kind") != "quantization"
        or quant.get("status") != "active_after_c_best"
        or quant.get("simulation_only") is not True
        or quant.get("targets") != ["C0", "C_best"]
        or quant.get("qat", {}).get("observer_update_epochs") != 3
    ):
        raise ConfigContractError(f"{quant_path}: quantization 契約漂移")

    for relative in catalog["datasets"].values():
        path = root / relative
        payload = load_yaml(path)
        validate_spec_metadata(payload, path, spec_path=root / "EXPERIMENT_SPEC.md")
        required = {
            "schema_version",
            "spec_version",
            "spec_sha256",
            "kind",
            "project_metadata",
            "path",
            "train",
            "val",
            "names",
        }
        optional = {
            "test",
            "nc",
            "kpt_shape",
            "flip_idx",
            "source_read_only",
            "group_key",
            "split_ratio",
            "split_seed",
        }
        metadata = payload.get("project_metadata")
        if (
            payload.get("schema_version") != 2
            or payload.get("kind") != "dataset"
            or not required <= set(payload)
            or set(payload) - required - optional
            or not isinstance(metadata, dict)
            or set(metadata)
            != {"title_zh", "summary_zh", "task", "ultralytics_dataset_yaml"}
            or metadata.get("ultralytics_dataset_yaml") is not True
        ):
            raise ConfigContractError(f"{path}: dataset 欄位不完整或未知")
        if payload.get("test") not in (None, ""):
            raise ConfigContractError(f"{path}: 來源沒有 test split，不得自行宣告")

    fusion_path = root / str(catalog["fusion_template"])
    fusion = load_yaml(fusion_path)
    validate_spec_metadata(fusion, fusion_path, spec_path=root / "EXPERIMENT_SPEC.md")
    source_fields = {
        "role",
        "checkpoint",
        "checkpoint_sha256",
        "architecture_yaml",
        "architecture_yaml_sha256",
        "head_contract",
    }
    fusion_fields = {
        "schema_version",
        "spec_version",
        "spec_sha256",
        "kind",
        "title_zh",
        "summary_zh",
        "status",
        "enabled",
        "task",
        "source_a",
        "source_b",
        "dataset_yaml",
        "dataset_yaml_sha256",
        "training_yaml_sha256",
        "fusion_mode",
        "switch_policy",
        "allowed_future_modes",
        "standalone_loadable",
    }
    if (
        set(fusion) != fusion_fields
        or fusion.get("schema_version") != 2
        or fusion.get("kind") != "fusion_source_pair_template"
        or fusion.get("status") != "future_disabled"
        or fusion.get("enabled") is not False
        or fusion.get("fusion_mode") is not None
        or fusion.get("switch_policy") is not None
        or fusion.get("standalone_loadable") is not False
        or set(fusion.get("source_a", {})) != source_fields
        or set(fusion.get("source_b", {})) != source_fields
        or fusion.get("allowed_future_modes")
        != ["weight_interpolation", "ensemble", "routed_switch", "staged_transfer"]
    ):
        raise ConfigContractError(f"{fusion_path}: 雙來源融合範本契約漂移")

    return ConfigCheckReport(
        True,
        SPEC_VERSION,
        file_sha256(root / "EXPERIMENT_SPEC.md"),
        ultralytics.__version__,
        "configs/catalog.yaml",
        tuple(sorted(candidates)),
        tuple(sorted(checked)),
        ("pose:rle（只有 Pose.flow_model 存在時才生效）",),
        tuple(sorted(RUNTIME_OVERRIDE_ALLOWLIST)),
        ("pose（必須由使用者明確啟用）", "fusion（目前停用，尚未決定模式）"),
    )


@dataclass(frozen=True)
class TrainingRecipe:
    data: str
    imgsz: int = 640
    batch: int = 16
    seed: int = 0
    optimizer: str = "MuSGD"
    device: str = "0"
    workers: int = 8
    amp: bool = True
    deterministic: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingRecipe:
        payload = load_yaml(path)
        source = payload.get("training", payload)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: source[key] for key in allowed if key in source})


def manifest_hashes(
    *,
    spec_path: Path,
    architecture_path: Path,
    training: EffectiveTrainingConfig,
    dataset_path: Path,
    parent_checkpoint: Path,
) -> dict[str, str]:
    """Return the mandatory immutable lineage hashes."""

    return {
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(spec_path),
        "architecture_yaml_sha256": file_sha256(architecture_path),
        "training_yaml_sha256": training.sha256,
        "formal_training_yaml_sha256": training.sha256,
        "dataset_yaml_sha256": file_sha256(dataset_path),
        "parent_checkpoint_sha256": file_sha256(parent_checkpoint),
    }


def write_effective_training(
    config: EffectiveTrainingConfig,
    destination: Path,
    *,
    args: dict[str, Any] | None = None,
) -> Path:
    """Materialize the exact forwarded arguments for audit and hashing."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "config_id": config.config_id,
        "title_zh": config.title_zh,
        "task": config.task,
        "candidate_id": config.candidate_id,
        "stage": config.stage,
        "formal_training_yaml": str(config.sources[0]),
        "formal_training_yaml_sha256": config.sha256,
        "training": dict(config.args if args is None else args),
    }
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination
