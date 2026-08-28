"""architecture_2 中文命令列介面。"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_json(value: Any, *, stream: Any | None = None) -> None:
    if is_dataclass(value):
        value = asdict(value)
    if stream is None:
        stream = sys.stdout
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), file=stream)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(value):
        value = asdict(value)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _load_callable(reference: str) -> Callable[..., Any]:
    module_ref, separator, function_name = reference.partition(":")
    if not separator or not module_ref or not function_name:
        raise ValueError("loader 格式必須是 module:function 或 /path/builder.py:function")
    if module_ref.endswith(".py") or "/" in module_ref:
        path = Path(module_ref).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        module_name = f"_achitechure_2_loader_{path.stem}"
        specification = importlib.util.spec_from_file_location(module_name, path)
        if specification is None or specification.loader is None:
            raise ImportError(f"無法載入 builder：{path}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    value = getattr(module, function_name, None)
    if not callable(value):
        raise TypeError(f"{reference} 不是 callable")
    return value


_FULL35_REVISION = "full35-final-j3-seed0-d67fb45c"
_CPU_GATE_FILES = {
    candidate: f"full35-{candidate.lower()}-dry-run.json" for candidate in ("C0", "C1", "C2", "C3")
}


def _float20_preflight(
    project_root: Path,
    *,
    source_cache_paths: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """唯讀驗證本機 materialized handoff、CPU候選與真實loss證據。"""

    root = project_root.resolve()
    cpu_root = root / "artifacts/cpu-validation"
    accepted_path = root / "artifacts/intake/accepted.json"
    required = {
        "accepted_handoff": accepted_path,
        **{
            f"cpu_{candidate.lower()}": cpu_root / filename for candidate, filename in _CPU_GATE_FILES.items()
        },
    }
    if source_cache_paths is None:
        source_cache_paths = (
            Path("/home/uxin/yolo/coco2017/labels/train2017.cache"),
            Path("/home/uxin/yolo/original/pose/derived/bbat5-v1/pose/labels/train.cache"),
        )
    missing = [name for name, path in required.items() if not path.is_file()]
    invalid: list[str] = []
    if any(path.exists() for path in source_cache_paths):
        invalid.append("source_adjacent_label_cache")
    accepted_revision: str | None = None

    if accepted_path.is_file():
        try:
            accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            invalid.append(f"accepted_handoff:{type(error).__name__}")
        else:
            accepted_revision = accepted.get("revision_id")
            if accepted.get("accepted") is not True or accepted_revision != _FULL35_REVISION:
                invalid.append("accepted_handoff:revision_or_status")

    passed_candidates: list[str] = []
    for candidate, filename in _CPU_GATE_FILES.items():
        path = cpu_root / filename
        if not path.is_file():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            build = report["build"]
            cpu = report["cpu_validation"]
            intake = report["handoff"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            invalid.append(f"cpu_{candidate.lower()}:{type(error).__name__}")
            continue
        if not (
            build.get("resolved_id") == candidate
            and intake.get("accepted") is True
            and intake.get("revision_id") == _FULL35_REVISION
            and cpu.get("passed") is True
            and cpu.get("geometry_imgsz") == 640
            and cpu.get("loss_is_finite") is True
            and cpu.get("gradients_are_finite") is True
            and cpu.get("state_dict_reload") is True
            and cpu.get("contract_unchanged") is True
            and cpu.get("frozen_gradient_count") == 0
        ):
            invalid.append(f"cpu_{candidate.lower()}:contract")
            continue
        passed_candidates.append(candidate)

    native_path = cpu_root / "full35-native-loss-smoke.json"
    native_loss_passed = False
    if native_path.is_file():
        try:
            native = json.loads(native_path.read_text(encoding="utf-8"))
            native_loss_passed = bool(
                native.get("schema_version") == 2
                and native.get("handoff", {}).get("revision_id") == _FULL35_REVISION
                and native.get("losses", {}).get("detect", {}).get("finite") is True
                and native.get("losses", {}).get("pose", {}).get("finite") is True
                and native.get("pose_rle", {}).get("active") is True
                and native.get("cache_policy", {}).get("source_adjacent_caches_absent") is True
            )
        except (OSError, json.JSONDecodeError, TypeError):
            native_loss_passed = False
        if not native_loss_passed:
            invalid.append("native_loss:contract")

    return {
        "ready": not missing and not invalid,
        "revision_id": accepted_revision,
        "passed_candidates": passed_candidates,
        "native_loss_passed": native_loss_passed,
        "missing": missing,
        "invalid": invalid,
        "source_adjacent_caches_absent": not any(path.exists() for path in source_cache_paths),
        "artifact_root": str(cpu_root),
    }


def _require_float20_preflight(project_root: Path) -> dict[str, Any]:
    report = _float20_preflight(project_root)
    if not report["ready"]:
        raise PermissionError(
            f"Float20 GPU preflight 未通過；missing={report['missing']} invalid={report['invalid']}"
        )
    return report


def _cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    from .config import SPEC_VERSION, file_sha256

    accepted = args.project_root / "artifacts/intake/accepted.json"
    data_root = Path("/home/uxin/yolo/original/pose/derived/bbat5-v1")
    screening_root = args.project_root / "artifacts/datasets/architecture-screen-20-v1"
    run_config = args.project_root / "configs/runs/full35-float-screen-20.yaml"
    spec_path = args.project_root / "EXPERIMENT_SPEC.md"
    handoff = None
    data_lineage: dict[str, Any] | None = None
    if accepted.is_file():
        payload = json.loads(accepted.read_text(encoding="utf-8"))
        handoff = payload.get("revision_id")
    rebuild_manifest = data_root / "manifests/rebuild-manifest.json"
    if not rebuild_manifest.is_file():
        rebuild_manifest = args.project_root / "artifacts/datasets/bbat5-v1/manifests/rebuild-manifest.json"
    if rebuild_manifest.is_file():
        payload = json.loads(rebuild_manifest.read_text(encoding="utf-8"))
        data_lineage = {
            "spec_version": payload.get("spec_version"),
            "spec_sha256": payload.get("spec_sha256"),
        }
    run_payload = yaml.safe_load(run_config.read_text(encoding="utf-8"))
    pose_decision = run_payload.get("authorization", {}).get("pose")
    pose_decided = isinstance(pose_decision, bool)
    preflight = _float20_preflight(args.project_root)
    return {
        "current_phase": "B" if handoff else "A",
        "readiness": (
            "ready_for_float20_queue"
            if preflight["ready"]
            else (
                "accepted_handoff_requires_candidate_cpu_validation"
                if handoff
                else "ready_for_upstream_handoff"
            )
        ),
        "authoritative_spec": str(spec_path),
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(spec_path),
        "fusion_winner": handoff or "waiting_for_yolo_combine_handoff",
        "bbat5_v1": "exists" if data_root.is_dir() else "not_built",
        "float_screen_20_manifests": (
            "exists" if (screening_root / "screening-manifest.json").is_file() else "not_built"
        ),
        "bbat5_v1_lineage": data_lineage,
        "float20_cpu_preflight": preflight,
        "pose_formal_execution": pose_decision,
        "gpu_actions": (
            "waiting_for_pose_decision"
            if not pose_decided
            else (
                "authorized_waiting_for_queue"
                if preflight["ready"]
                else "authorized_waiting_for_cpu_preflight"
            )
        ),
        "formal_training": (
            "ready_to_queue"
            if preflight["ready"] and pose_decided
            else (
                "waiting_for_pose_decision" if preflight["ready"] else "waiting_for_handoff_or_cpu_preflight"
            )
        ),
        "說明": "status 不會啟動訓練、不會查占用中的 GPU，也不會自動選 C_best。",
    }


def _cmd_config_check(args: argparse.Namespace) -> dict[str, Any]:
    from .config import check_configs

    return check_configs(args.project_root).to_dict()


def _cmd_show_candidates(_: argparse.Namespace) -> dict[str, Any]:
    from .config import CANDIDATES

    descriptions = {
        "C0": "Fusion Winner 原樣 reference；C0-Control 使用同等恢復預算。",
        "C1": "只把所選 C3k2 hidden expansion e：0.5 → 0.375。",
        "C2": "只把所選 C3k2 inner bottleneck 數量：2 → 1。",
        "C3": "只把所選 C3k2 第一個 inner kernel：3×3 → 1×1。",
    }
    return {
        "target_resolution": "Full35 J3已驗證shared paths：graph.model.6/8/13/19",
        "combination_policy": "第一輪禁止組合；C3-P5/R1 等使用者看完結果再決定",
        "candidates": {
            candidate_id: {
                "說明": descriptions[candidate_id],
                "factors": {
                    "e": spec.factors[0],
                    "inner_n": spec.factors[1],
                    "kernel_mode": spec.factors[2],
                    "use_rep": spec.factors[3],
                },
                "changed_fields": list(spec.changed_fields),
                "quantization_eligible": spec.quantization_eligible,
            }
            for candidate_id, spec in CANDIDATES.items()
        },
    }


def _cmd_prepare_pose_data(args: argparse.Namespace) -> dict[str, Any]:
    from .pose_data import prepare_bbat5_dataset

    return prepare_bbat5_dataset(
        args.pose_source,
        args.detect_source,
        args.destination,
        coco_train_list=args.coco_train_list,
        train_ratio=args.train_ratio,
        search_val_ratio=args.search_val_ratio,
        seed=args.seed,
        execute=args.execute,
        expected_patch_count=args.expected_patch_count,
    ).to_dict()


def _cmd_validate_pose_data(args: argparse.Namespace) -> dict[str, Any]:
    from .pose_data import validate_bbat5_dataset

    return validate_bbat5_dataset(args.destination)


def _cmd_prepare_screening_data(args: argparse.Namespace) -> dict[str, Any]:
    from .screening_data import prepare_screening_data

    return prepare_screening_data(
        coco_train_list=args.coco_train_list,
        bbat5_pose_search_train=args.bbat5_pose_search_train,
        bbat5_detect_search_train=args.bbat5_detect_search_train,
        bbat5_pose_search_val=args.bbat5_pose_search_val,
        destination=args.destination,
        fraction=args.fraction,
        coco_search_val_size=args.coco_search_val_size,
        seed=args.seed,
        execute=args.execute,
    ).to_dict()


def _cmd_validate_screening_data(args: argparse.Namespace) -> dict[str, Any]:
    from .screening_data import validate_screening_data

    return validate_screening_data(args.destination)


def _cmd_export_data_metadata(args: argparse.Namespace) -> dict[str, Any]:
    from .pose_data import export_bbat5_metadata

    return export_bbat5_metadata(
        args.source,
        args.destination,
        execute=args.execute,
    )


def _cmd_export_github_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from .pose_data import export_bbat5_github_dataset

    return export_bbat5_github_dataset(
        args.source,
        args.destination,
        execute=args.execute,
    )


def _cmd_validate_github_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from .pose_data import validate_bbat5_github_dataset

    return validate_bbat5_github_dataset(args.destination)


def _cmd_inspect_handoff(args: argparse.Namespace) -> dict[str, Any]:
    from .intake import validate_handoff

    report = validate_handoff(
        args.manifest,
        project_root=args.project_root,
        loader=None,
    )
    payload = asdict(report)
    payload["inspection_only"] = True
    payload["accepted"] = False
    return payload


def _cmd_accept_handoff(args: argparse.Namespace) -> dict[str, Any]:
    from .intake import validate_handoff, write_intake

    loader = _load_callable(args.model_loader)
    report = validate_handoff(
        args.manifest,
        project_root=args.project_root,
        loader=loader,
    )
    destination = args.output or args.project_root / "artifacts/intake/accepted.json"
    write_intake(report, destination)
    payload = asdict(report)
    payload["accepted_artifact"] = str(destination.resolve())
    return payload


def _cmd_full35_fresh_process(args: argparse.Namespace) -> dict[str, Any]:
    from .full35_adapter import build_full35_fresh_process_report

    report = build_full35_fresh_process_report(args.full35_root, imgsz=args.imgsz)
    if args.output:
        _write_json(args.output, report)
        report["output"] = str(args.output.resolve())
    return report


def _cmd_native_loss_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from .native_loss_smoke import run_cpu_native_loss_smoke

    report = run_cpu_native_loss_smoke(args.config)
    _write_json(args.output, report)
    return {
        **report,
        "output": str(args.output.resolve()),
    }


def _cmd_resolve_candidates(args: argparse.Namespace) -> dict[str, Any]:
    from .candidate import resolve_candidate_matrix
    from .intake import HandoffManifest

    manifest = HandoffManifest.load(args.manifest)
    matrix = resolve_candidate_matrix(manifest.fusion_kind, manifest.candidate_regions)
    return {
        "handoff_revision": manifest.revision_id,
        "winner_id": manifest.winner_id,
        "fusion_kind": manifest.fusion_kind,
        "resolved_candidates": [
            {
                "base_candidate": item.base_candidate_id,
                "resolved_candidate": item.resolved_id,
                "region_id": item.region.region_id if item.region else None,
                "region_role": item.region.role if item.region else None,
                "module_paths": list(item.region.module_paths) if item.region else [],
            }
            for item in matrix
        ],
    }


def _cmd_cpu_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    from .candidate import build_candidate, resolve_candidate_matrix
    from .cpu_validation import validate_cpu_candidate
    from .graph import inspect_fusion_graph
    from .intake import HandoffManifest, validate_handoff

    loader = _load_callable(args.model_loader)
    materialized: list[Any] = []

    def capture(checkpoint: Path) -> Any:
        model = loader(checkpoint)
        materialized.append(model)
        return model

    intake = validate_handoff(
        args.manifest,
        project_root=args.project_root,
        loader=capture,
    )
    manifest = HandoffManifest.load(args.manifest)
    matrix = resolve_candidate_matrix(manifest.fusion_kind, manifest.candidate_regions)
    matches = [item for item in matrix if item.resolved_id == args.candidate]
    if len(matches) != 1:
        available = [item.resolved_id for item in matrix]
        raise ValueError(f"找不到唯一 resolved candidate {args.candidate!r}；可用 {available}")
    resolved = matches[0]
    parent = materialized[0]
    candidate, build = build_candidate(parent, resolved, seed=args.seed)
    graph = inspect_fusion_graph(
        candidate,
        fusion_kind=manifest.fusion_kind,
        candidate_regions=manifest.candidate_regions,
        protected_module_paths=manifest.protected_module_paths,
        frozen_module_paths=manifest.frozen_module_paths,
        expected_candidate=build,
    )

    def builder() -> Any:
        fresh_parent = loader(manifest.checkpoint.path)
        return build_candidate(fresh_parent, resolved, seed=args.seed)[0]

    cpu = validate_cpu_candidate(
        candidate,
        builder=builder,
        frozen_module_paths=manifest.frozen_module_paths,
        smoke_imgsz=args.smoke_imgsz,
        geometry_imgsz=args.geometry_imgsz,
    )
    payload = {
        "handoff": asdict(intake),
        "build": build.to_dict(),
        "graph": graph.to_dict(),
        "cpu_validation": cpu.to_dict(),
    }
    if args.output:
        _write_json(args.output, payload)
        payload["output"] = str(args.output.resolve())
    return payload


def _scalar(value: str) -> Any:
    return yaml.safe_load(value)


def _cmd_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    from .config import resolve_training_template, write_effective_training
    from .intake import HandoffManifest, validate_handoff

    validate_handoff(args.manifest, project_root=args.project_root, loader=None)
    manifest = HandoffManifest.load(args.manifest)
    recipe = yaml.safe_load(manifest.training_recipe.path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise TypeError("handoff training recipe 必須是 mapping")
    overrides = {
        name: value
        for name, value in {
            "name": args.name,
            "project": str(args.project) if args.project else None,
            "device": args.device,
            "workers": args.workers,
            "cache": _scalar(args.cache) if args.cache is not None else None,
        }.items()
        if value is not None
    }
    config = resolve_training_template(
        args.template,
        handoff_recipe=recipe,
        runtime_overrides=overrides,
        project_root=args.project_root,
    )
    payload = {
        "config_id": config.config_id,
        "training": config.args,
        "source_yaml_sha256": config.sha256,
        "formal_execution": "仍需 --enable-pose 與 GPU 授權；此命令只解析設定",
    }
    if args.output:
        write_effective_training(config, args.output)
        payload["output"] = str(args.output.resolve())
    return payload


def _cmd_float20_plan(args: argparse.Namespace) -> dict[str, Any]:
    """讀取正式run YAML與本機preflight；不會啟動GPU。"""

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Float20 run YAML 必須是 mapping")
    training = payload.get("training")
    authorization = payload.get("authorization")
    queue = payload.get("queue")
    if not all(isinstance(value, dict) for value in (training, authorization, queue)):
        raise TypeError("Float20 run YAML 缺少 training/authorization/queue mapping")
    pose = authorization["pose"]
    preflight = _float20_preflight(args.project_root)
    blockers: list[str] = []
    if not isinstance(pose, bool):
        blockers.append("請先把authorization.pose與training.pose_enabled同時設為true或false")
    if not preflight["ready"]:
        blockers.append(f"CPU preflight未通過：missing={preflight['missing']} invalid={preflight['invalid']}")
    return {
        "config": str(args.config.resolve()),
        "status": payload.get("status"),
        "candidates": payload.get("candidates"),
        "gpu_authorized": authorization.get("gpu"),
        "pose_decision": pose,
        "cpu_preflight": preflight,
        "ready_to_queue": not blockers,
        "blocker": "；".join(blockers) if blockers else None,
        "batch": training.get("batch"),
        "fraction": training.get("fraction"),
        "scale": training.get("scale"),
        "cache": training.get("cache"),
        "epochs": training.get("epochs"),
        "queue": queue,
        "automatic_post_float_quantization": False,
    }


def _require_queue_grant(
    args: argparse.Namespace,
    *,
    queue_name: str | None = None,
) -> dict[str, Any]:
    """只接受由持鎖 yolo_combine queue 啟動的當前 child process。"""

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queue"), dict):
        raise TypeError("run YAML 缺少 queue mapping")
    queue_root = payload["queue"]
    queue = queue_root
    if queue_name is not None:
        queue = queue_root.get(queue_name)
        if not isinstance(queue, dict):
            raise TypeError(f"run YAML 缺少 queue.{queue_name} mapping")
    expected_state = Path(str(queue["state"]))
    if not expected_state.is_absolute():
        expected_state = args.project_root / expected_state
    expected_state = expected_state.resolve()
    supplied_state = args.queue_state.resolve()
    if supplied_state != expected_state:
        raise PermissionError(f"queue grant state 路徑不符：{supplied_state} != {expected_state}")

    parent_cmdline_path = Path(f"/proc/{os.getppid()}/cmdline")
    try:
        parent_cmdline = parent_cmdline_path.read_bytes().replace(b"\x00", b" ").decode()
    except OSError as error:
        raise PermissionError("無法驗證 GPU queue parent process") from error
    if "yolo_combine.gpu_queue" not in parent_cmdline:
        raise PermissionError("GPU run 只能由 yolo_combine.gpu_queue 啟動")

    state: dict[str, Any] | None = None
    for _ in range(100):
        if expected_state.is_file():
            candidate = json.loads(expected_state.read_text(encoding="utf-8"))
            if (
                isinstance(candidate, dict)
                and candidate.get("status") == "running"
                and candidate.get("child_pid") == os.getpid()
            ):
                state = candidate
                break
        time.sleep(0.1)
    if state is None:
        raise PermissionError("GPU queue 未授予此 process 執行權；拒絕直接啟動")
    if state.get("queue_id") != queue.get("queue_id"):
        raise PermissionError("GPU queue_id 與正式 run YAML 不符")

    lock_path = Path(str(queue["lock"]))
    if not lock_path.is_absolute():
        lock_path = args.project_root / lock_path
    if not lock_path.is_file():
        raise PermissionError("GPU queue lock 不存在；拒絕執行")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(queue["gpu_index"])
    return state


def _cmd_float20_run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise PermissionError("Float20 真實執行必須明確提供 --execute")
    _require_queue_grant(args)
    plan = _cmd_float20_plan(args)
    if not plan["ready_to_queue"]:
        raise PermissionError(f"Float20計畫尚未ready：{plan['blocker']}")

    from .config import check_configs

    check_configs(args.project_root)
    from .screen_training import run_screen_matrix

    return run_screen_matrix(args.config, execute=True)



def _cmd_full_run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise PermissionError("C2/C3完整訓練必須明確提供 --execute")
    _require_queue_grant(args, queue_name="full_training")
    from .full_training import run_full_matrix

    return run_full_matrix(args.config, execute=True)


def _cmd_quant_run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise PermissionError("PTQ/QAT-lite必須明確提供 --execute")
    _require_queue_grant(args, queue_name="quantization")
    from .quant_training import run_quant_matrix

    return run_quant_matrix(args.config, execute=True)

def _cmd_export_downstream_results(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise PermissionError("最終接續結果匯出必須明確提供 --execute")
    from .downstream_export import export_downstream_results

    return export_downstream_results(args.config, output=args.output)
def _cmd_float20_profile(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise PermissionError("Float20 正式成本 profile 必須明確提供 --execute")
    _require_queue_grant(args)
    from .profiling import profile_float20_candidates

    return profile_float20_candidates(
        args.config,
        output=args.output,
        warmup=args.warmup,
        iterations=args.iterations,
    )


def _cmd_export_float20_results(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise PermissionError("Float20 正式結果匯出必須明確提供 --execute")
    from .result_export import export_float20_results

    return export_float20_results(
        args.config,
        profiles_path=args.profiles,
        output_dir=args.output,
    )


def _cmd_queue_status(args: argparse.Namespace) -> dict[str, Any]:
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queue"), dict):
        raise TypeError("Float20 run YAML 缺少 queue mapping")
    state = Path(str(payload["queue"]["state"]))
    if not state.is_absolute():
        state = args.project_root / state
    if not state.is_file():
        return {
            "status": "not_launched",
            "state": str(state.resolve()),
            "pose_decision": payload.get("authorization", {}).get("pose"),
        }
    value = json.loads(state.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("queue state 必須是 JSON object")
    return value


def _candidate_metrics(value: dict[str, Any]) -> Any:
    from .decisions import CandidateMetrics, ClassMetrics

    payload = dict(value)
    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise TypeError("metrics.classes 必須是 ball/bat mapping")
    payload["classes"] = {name: ClassMetrics(**class_payload) for name, class_payload in classes.items()}
    return CandidateMetrics(**payload)


def _cmd_assess(args: argparse.Namespace) -> dict[str, Any]:
    from .decisions import evaluate_float_results

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    entries = raw.get("candidates") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise TypeError("結果 JSON 必須是 list 或含 candidates list")
    report = evaluate_float_results(_candidate_metrics(item) for item in entries)
    payload = report.to_dict()
    if args.output:
        _write_json(args.output, payload)
        payload["output"] = str(args.output.resolve())
    return payload


def _cmd_quant_check(args: argparse.Namespace) -> dict[str, Any]:
    from .quantization import require_quantization_stage

    status = require_quantization_stage(
        args.candidate,
        args.stage,
        user_approved=args.approved,
        gpu_authorized=args.gpu_authorized,
    )
    return {
        "candidate": args.candidate,
        "stage": args.stage.upper(),
        "status": status,
        "simulation_only": True,
        "執行": "此命令只檢查資格，不會開始 PTQ/QAT。",
    }


def _cmd_qat_lite_check(args: argparse.Namespace) -> dict[str, Any]:
    from .qat_lite import require_qat_lite_stage

    status = require_qat_lite_stage(
        args.candidate,
        user_approved=args.approved,
        gpu_authorized=args.gpu_authorized,
    )
    return {
        "candidate": args.candidate,
        "stage": "Q2L",
        "status": status,
        "simulation_only": True,
        "執行": "此命令只檢查 QAT-lite 資格，不會開始訓練。",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m achitechure_2",
        description="architecture_2：融合 winner 上的 C0/C1/C2/C3 評估工具",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="architecture_2 專案根目錄",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="顯示 Phase A、Pose/GPU 與 handoff 狀態")
    commands.add_parser("config-check", help="驗證全部正式 YAML/schema/spec hashes")
    commands.add_parser("show-candidates", help="以中文說明 C0/C1/C2/C3")

    prepare = commands.add_parser(
        "prepare-pose-data",
        aliases=["prepare-bbat5-data"],
        help="規劃或建立 BBAT5 v1；不會啟動 Pose",
    )
    prepare.add_argument("--pose-source", type=Path, default=Path("/home/uxin/yolo/original/pose/dataset"))
    prepare.add_argument(
        "--detect-source",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/detect_dataset"),
    )
    prepare.add_argument(
        "--destination",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/derived/bbat5-v1"),
    )
    prepare.add_argument(
        "--coco-train-list",
        type=Path,
        default=Path("/home/uxin/yolo/coco2017/train2017.txt"),
    )
    prepare.add_argument("--train-ratio", type=float, default=0.9)
    prepare.add_argument("--search-val-ratio", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--expected-patch-count", type=int, default=4)
    prepare.add_argument("--execute", action="store_true", help="實際建立 immutable 衍生版本")

    screen = commands.add_parser(
        "prepare-screening-data",
        help="規劃或建立固定20%%架構篩選 manifests；不複製影像",
    )
    screen.add_argument(
        "--coco-train-list",
        type=Path,
        default=Path("/home/uxin/yolo/coco2017/train2017.txt"),
    )
    screen.add_argument(
        "--bbat5-pose-search-train",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/derived/bbat5-v1/pose/splits/search-train.txt"),
    )
    screen.add_argument(
        "--bbat5-detect-search-train",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/derived/bbat5-v1/detect/splits/search-train.txt"),
    )
    screen.add_argument(
        "--bbat5-pose-search-val",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/derived/bbat5-v1/pose/splits/search-val.txt"),
    )
    screen.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "artifacts/datasets/architecture-screen-20-v1",
    )
    screen.add_argument("--fraction", type=float, default=0.2)
    screen.add_argument("--coco-search-val-size", type=int, default=5000)
    screen.add_argument("--seed", type=int, default=0)
    screen.add_argument("--execute", action="store_true", help="實際建立不可覆寫的 manifest View")

    screen_validate = commands.add_parser(
        "validate-screening-data",
        help="驗證20%% manifests、任務 assignment 與 leakage",
    )
    screen_validate.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "artifacts/datasets/architecture-screen-20-v1",
    )

    validate = commands.add_parser("validate-pose-data", help="驗證已建立的 BBAT5 v1")
    validate.add_argument(
        "--destination",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/derived/bbat5-v1"),
    )

    export = commands.add_parser(
        "export-data-metadata",
        help="只匯出 README/YAML/manifests 到 Git；不含影像或 labels",
    )
    export.add_argument(
        "--source",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/derived/bbat5-v1"),
    )
    export.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "artifacts/datasets/bbat5-v1",
    )
    export.add_argument("--execute", action="store_true")

    github_export = commands.add_parser(
        "export-github-dataset",
        help="物化完整可攜 BBAT5 v1；含影像/labels，不含 weights",
    )
    github_export.add_argument(
        "--source",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/derived/bbat5-v1"),
    )
    github_export.add_argument(
        "--destination",
        type=Path,
        default=(DEFAULT_PROJECT_ROOT / "artifacts/datasets/bbat5-v1/github-dataset"),
    )
    github_export.add_argument(
        "--execute",
        action="store_true",
        help="實際建立不可覆寫 snapshot；省略時只顯示計畫",
    )

    github_validate = commands.add_parser(
        "validate-github-dataset",
        help="驗證 GitHub dataset 無 symlink/權重且 split 可攜",
    )
    github_validate.add_argument(
        "--destination",
        type=Path,
        default=(DEFAULT_PROJECT_ROOT / "artifacts/datasets/bbat5-v1/github-dataset"),
    )

    inspect = commands.add_parser("inspect-handoff", help="只驗證 handoff metadata；不寫 accepted")
    inspect.add_argument("--manifest", type=Path, required=True)

    fresh = commands.add_parser(
        "full35-fresh-process",
        help="在獨立 CPU 程序嚴格載入 accepted J3 並驗證 task=both forward",
    )
    fresh.add_argument(
        "--full35-root",
        type=Path,
        default=Path("/home/uxin/yolo/yolo_combine/final/full35"),
    )
    fresh.add_argument("--imgsz", type=int, default=64)
    fresh.add_argument("--output", type=Path)

    native_loss = commands.add_parser(
        "native-loss-smoke",
        help="以 CPU real-data batch 驗證 J3 Detect/PoseLoss26/RLE loss",
    )
    native_loss.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "configs/runs/full35-float-screen-20.yaml",
    )
    native_loss.add_argument("--output", type=Path, required=True)

    accept = commands.add_parser("accept-handoff", help="materialize graph 後寫入 accepted intake")
    accept.add_argument("--manifest", type=Path, required=True)
    accept.add_argument("--model-loader", required=True, help="module:function 或 builder.py:function")
    accept.add_argument("--output", type=Path)

    resolve = commands.add_parser("resolve-candidates", help="依 fusion kind 顯示實際候選矩陣")
    resolve.add_argument("--manifest", type=Path, required=True)

    dry_run = commands.add_parser("cpu-dry-run", help="對一個 resolved candidate 做 CPU 完整 smoke")
    dry_run.add_argument("--manifest", type=Path, required=True)
    dry_run.add_argument("--model-loader", required=True)
    dry_run.add_argument("--candidate", default="C0")
    dry_run.add_argument("--seed", type=int, default=0)
    dry_run.add_argument("--smoke-imgsz", type=int, default=64)
    dry_run.add_argument("--geometry-imgsz", type=int, default=640)
    dry_run.add_argument("--output", type=Path)

    effective = commands.add_parser(
        "effective-config",
        help="由 handoff 配方解析完整 training YAML；不執行訓練",
    )
    effective.add_argument("--manifest", type=Path, required=True)
    effective.add_argument("--template", type=Path, default=Path("configs/training/float-main.yaml"))
    effective.add_argument("--name")
    effective.add_argument("--project", type=Path)
    effective.add_argument("--device")
    effective.add_argument("--workers", type=int)
    effective.add_argument("--cache")
    effective.add_argument("--output", type=Path)

    assess = commands.add_parser("assess", help="產生 measurement-first Float/Pareto 報告")
    assess.add_argument("--input", type=Path, required=True)
    assess.add_argument("--output", type=Path)

    quant = commands.add_parser("quant-check", help="只檢查候選 Q0/Q1/Q2 資格")
    quant.add_argument("--candidate", required=True)
    quant.add_argument("--stage", choices=["Q0", "Q1", "Q2", "q0", "q1", "q2"], required=True)
    quant.add_argument("--approved", action="append", default=[])
    quant.add_argument("--gpu-authorized", action="store_true")

    qat_lite = commands.add_parser(
        "qat-lite-check",
        help="只檢查候選 Q2L 短 QAT 資格，不啟動訓練",
    )
    qat_lite.add_argument("--candidate", required=True)
    qat_lite.add_argument("--approved", action="append", default=[])
    qat_lite.add_argument("--gpu-authorized", action="store_true")

    float20_plan = commands.add_parser(
        "float20-plan",
        help="顯示 Full35 C0～C3 Float20 計畫與 Pose gate；不啟動訓練",
    )
    float20_plan.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "configs/runs/full35-float-screen-20.yaml",
    )

    float20_run = commands.add_parser(
        "float20-run",
        help="僅供持鎖 GPU 佇列執行 Full35 C0～C3 Float20 矩陣",
    )
    float20_run.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "configs/runs/full35-float-screen-20.yaml",
    )
    float20_run.add_argument("--queue-state", type=Path, required=True, help=argparse.SUPPRESS)
    float20_run.add_argument(
        "--execute",
        action="store_true",
        help="確認執行正式 Float20 矩陣；仍須通過 queue grant",
    )

    downstream = DEFAULT_PROJECT_ROOT / "configs/runs/full35-c2-c3-auto-continuation.yaml"
    full_run = commands.add_parser(
        "full-run",
        help="僅供持鎖GPU佇列執行合格C2/C3完整資料訓練",
    )
    full_run.add_argument("--config", type=Path, default=downstream)
    full_run.add_argument(
        "--queue-state",
        type=Path,
        required=True,
        help=argparse.SUPPRESS,
    )
    full_run.add_argument("--execute", action="store_true")

    quant_run = commands.add_parser(
        "quant-run",
        help="僅供持鎖GPU佇列執行C2/C3的Q0/PTQ/QAT-lite",
    )
    quant_run.add_argument("--config", type=Path, default=downstream)
    quant_run.add_argument(
        "--queue-state",
        type=Path,
        required=True,
        help=argparse.SUPPRESS,
    )
    quant_run.add_argument("--execute", action="store_true")
    downstream_export = commands.add_parser(
        "export-downstream-results",
        help="匯出完整訓練、PTQ、QAT-lite與C_best中文報告",
    )
    downstream_export.add_argument("--config", type=Path, default=downstream)
    downstream_export.add_argument("--output", type=Path)
    downstream_export.add_argument("--execute", action="store_true")

    float20_profile = commands.add_parser(
        "float20-profile",
        help="僅供持鎖 GPU 佇列在完整矩陣後量測 Params／GFLOPs／latency／VRAM",
    )
    float20_profile.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "configs/runs/full35-float-screen-20.yaml",
    )
    float20_profile.add_argument("--queue-state", type=Path, required=True, help=argparse.SUPPRESS)
    float20_profile.add_argument("--output", type=Path)
    float20_profile.add_argument("--warmup", type=int, default=25)
    float20_profile.add_argument("--iterations", type=int, default=100)
    float20_profile.add_argument("--execute", action="store_true")

    float20_export = commands.add_parser(
        "export-float20-results",
        help="完整矩陣與成本量測後匯出 JSON／CSV／中文報告／圖表",
    )
    float20_export.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "configs/runs/full35-float-screen-20.yaml",
    )
    float20_export.add_argument(
        "--profiles",
        type=Path,
        help="cost-profiles.json；省略時使用 run root 的正式路徑",
    )
    float20_export.add_argument("--output", type=Path)
    float20_export.add_argument("--execute", action="store_true")

    queue_status = commands.add_parser(
        "queue-status",
        help="讀取 Float20 GPU 佇列狀態；不啟動或變更服務",
    )
    queue_status.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "configs/runs/full35-float-screen-20.yaml",
    )
    return parser


_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "status": _cmd_status,
    "config-check": _cmd_config_check,
    "show-candidates": _cmd_show_candidates,
    "prepare-pose-data": _cmd_prepare_pose_data,
    "prepare-bbat5-data": _cmd_prepare_pose_data,
    "validate-pose-data": _cmd_validate_pose_data,
    "prepare-screening-data": _cmd_prepare_screening_data,
    "validate-screening-data": _cmd_validate_screening_data,
    "export-data-metadata": _cmd_export_data_metadata,
    "export-github-dataset": _cmd_export_github_dataset,
    "validate-github-dataset": _cmd_validate_github_dataset,
    "inspect-handoff": _cmd_inspect_handoff,
    "full35-fresh-process": _cmd_full35_fresh_process,
    "native-loss-smoke": _cmd_native_loss_smoke,
    "accept-handoff": _cmd_accept_handoff,
    "resolve-candidates": _cmd_resolve_candidates,
    "cpu-dry-run": _cmd_cpu_dry_run,
    "effective-config": _cmd_effective_config,
    "assess": _cmd_assess,
    "quant-check": _cmd_quant_check,
    "qat-lite-check": _cmd_qat_lite_check,
    "full-run": _cmd_full_run,
    "quant-run": _cmd_quant_run,
    "export-downstream-results": _cmd_export_downstream_results,
    "float20-plan": _cmd_float20_plan,
    "float20-run": _cmd_float20_run,
    "float20-profile": _cmd_float20_profile,
    "export-float20-results": _cmd_export_float20_results,
    "queue-status": _cmd_queue_status,
}


def main(argv: list[str] | None = None) -> int:
    # 所有命令先隱藏CUDA；GPU run僅在queue grant驗證後、import torch前解鎖。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = build_parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    try:
        payload = _HANDLERS[args.command](args)
    except Exception as error:  # noqa: BLE001 - CLI 邊界需把所有失敗轉成結構化 JSON。
        _print_json(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            stream=sys.stderr,
        )
        return 2
    _print_json(payload)
    return 0
