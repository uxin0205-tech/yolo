"""architecture_2 Phase A 中文命令列介面。"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
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


def _cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    from .config import SPEC_VERSION, file_sha256

    accepted = args.project_root / "artifacts/intake/accepted.json"
    data_root = Path("/home/uxin/yolo/original/pose/derived/bbat5-v1")
    spec_path = args.project_root / "EXPERIMENT_SPEC.md"
    handoff = None
    data_lineage: dict[str, Any] | None = None
    if accepted.is_file():
        payload = json.loads(accepted.read_text(encoding="utf-8"))
        handoff = payload.get("revision_id")
    rebuild_manifest = data_root / "manifests/rebuild-manifest.json"
    if rebuild_manifest.is_file():
        payload = json.loads(rebuild_manifest.read_text(encoding="utf-8"))
        data_lineage = {
            "spec_version": payload.get("spec_version"),
            "spec_sha256": payload.get("spec_sha256"),
        }
    return {
        "current_phase": "B" if handoff else "A",
        "readiness": (
            "accepted_handoff_requires_candidate_cpu_validation"
            if handoff
            else "ready_for_upstream_handoff"
        ),
        "authoritative_spec": str(spec_path),
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(spec_path),
        "fusion_winner": handoff or "waiting_for_yolo_combine_handoff",
        "bbat5_v1": "exists" if data_root.is_dir() else "not_built",
        "bbat5_v1_lineage": data_lineage,
        "pose_formal_execution": "requires_user_opt_in",
        "gpu_actions": "blocked_until_user_authorization",
        "formal_training": "blocked_until_handoff_and_gpu_authorization",
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
        "target_resolution": "由 yolo_combine handoff 的 Candidate Regions 動態解析",
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


def _candidate_metrics(value: dict[str, Any]) -> Any:
    from .decisions import CandidateMetrics, ClassMetrics

    payload = dict(value)
    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise TypeError("metrics.classes 必須是 ball/bat mapping")
    payload["classes"] = {
        name: ClassMetrics(**class_payload) for name, class_payload in classes.items()
    }
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

    validate = commands.add_parser("validate-pose-data", help="驗證已建立的 BBAT5 v1")
    validate.add_argument(
        "--destination",
        type=Path,
        default=Path("/home/uxin/yolo/original/pose/derived/bbat5-v1"),
    )


    inspect = commands.add_parser("inspect-handoff", help="只驗證 handoff metadata；不寫 accepted")
    inspect.add_argument("--manifest", type=Path, required=True)

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
    return parser


_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "status": _cmd_status,
    "config-check": _cmd_config_check,
    "show-candidates": _cmd_show_candidates,
    "prepare-pose-data": _cmd_prepare_pose_data,
    "prepare-bbat5-data": _cmd_prepare_pose_data,
    "validate-pose-data": _cmd_validate_pose_data,
    "inspect-handoff": _cmd_inspect_handoff,
    "accept-handoff": _cmd_accept_handoff,
    "resolve-candidates": _cmd_resolve_candidates,
    "cpu-dry-run": _cmd_cpu_dry_run,
    "effective-config": _cmd_effective_config,
    "assess": _cmd_assess,
    "quant-check": _cmd_quant_check,
}


def main(argv: list[str] | None = None) -> int:
    # 本 revision 的 CLI 沒有正式 GPU 執行命令；所有可執行 smoke 強制隱藏 CUDA。
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
