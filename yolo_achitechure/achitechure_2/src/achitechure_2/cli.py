"""Dry-run-first command surface for the Stage-Aware Lite-C3k2 workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import CANDIDATES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ChineseArgumentParser(argparse.ArgumentParser):
    """以繁體中文顯示 argparse 的固定介面文字。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "位置參數"
        self._optionals.title = "選項"
        self.add_argument("-h", "--help", action="help", help="顯示此說明並離開")

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法:", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法:", 1)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseArgumentParser(prog="python -m achitechure_2", description="architecture_2 可重現實驗工具")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="專案根目錄")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("preflight", help="檢查執行環境、設定、CUDA 與上游交付門檻")
    commands.add_parser("config-check", help="嚴格檢查規格、YAML 與 Ultralytics 相容性")

    pose_data = commands.add_parser("prepare-pose-data", help="預覽或建立來源群組隔離的 Pose 資料集")
    pose_data.add_argument("--source", type=Path, default=Path("../../original/pose/bbt5.v1i.yolov8"))
    pose_data.add_argument("--destination", type=Path, default=Path("artifacts/datasets/pose_grouped"))
    pose_data.add_argument("--execute", action="store_true")

    intake = commands.add_parser("intake", help="驗證 architecture_1 正式交付資料")
    intake.add_argument("--manifest", type=Path, required=True)
    intake.add_argument("--execute", action="store_true")

    graph = commands.add_parser("inspect", help="檢查 checkpoint 模型圖")
    graph.add_argument("--checkpoint", type=Path, required=True)
    graph.add_argument("--allow-missing-masf", action="store_true")

    build = commands.add_parser("build", help="建立互相獨立的架構候選")
    build.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    build.add_argument("--output", type=Path)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--execute", action="store_true")

    pose_build = commands.add_parser("build-pose", help="將本機官方 Pose26 head 接到 C0/C_best；預設不執行")
    pose_build.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    pose_build.add_argument("--checkpoint", type=Path, required=True)
    pose_build.add_argument("--output", type=Path)
    pose_build.add_argument("--seed", type=int, default=0)
    pose_build.add_argument("--enable-pose", action="store_true", help="明確同意進入 Pose 路線")
    pose_build.add_argument("--execute", action="store_true")

    train = commands.add_parser("train", help="執行一個受門檻保護的訓練階段")
    train.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument(
        "--config",
        type=Path,
        help="推薦：指定一份 configs/training 下的完整正式 YAML",
    )
    train.add_argument("--task", choices=("detect", "pose"), help="相容介面；未用 --config 時預設 detect")
    train.add_argument(
        "--stage",
        choices=("D0", "D1", "D2", "P0", "P1", "P2", "P3", "P4", "Q2", "smoke", "formal", "extension", "qat"),
        help="相容介面；推薦改用 --config",
    )
    train.add_argument("--run-id", required=True)
    train.add_argument("--smoke-epochs", type=int, default=3)
    train.add_argument("--enable-pose", action="store_true", help="當 task=pose 時明確同意執行")
    train.add_argument("--execute", action="store_true")

    extend = commands.add_parser("extension-gate")
    extend.add_argument("--metrics", type=Path, required=True, help="含 100 個 mAP50-95 值的 JSON 陣列")
    extend.add_argument("--best-epoch", type=int, required=True)
    extend.add_argument("--early-stopped", action="store_true")

    assess = commands.add_parser("assess")
    assess.add_argument("--c0", type=Path, required=True)
    assess.add_argument("--candidate", type=Path, action="append", required=True)
    assess.add_argument("--r1-fusion-report", type=Path)
    assess.add_argument("--execute", action="store_true", help="寫入 artifacts/selection.json")

    rep = commands.add_parser("rep-fusion", help="驗證 R1 RepConv 融合前後等價性")
    rep.add_argument("--checkpoint", type=Path, required=True)
    rep.add_argument("--output", type=Path, required=True)
    rep.add_argument("--execute", action="store_true")

    fused = commands.add_parser("fuse-reference", help="建立 Q0 融合後 FP32 參考模型")
    fused.add_argument("--candidate", required=True)
    fused.add_argument("--checkpoint", type=Path, required=True)
    fused.add_argument("--output", type=Path, required=True)
    fused.add_argument("--execute", action="store_true")

    quant = commands.add_parser("quant-prepare", help="準備 Conv 假量化模擬")
    quant.add_argument("--candidate", required=True)
    quant.add_argument("--checkpoint", type=Path, required=True)
    quant.add_argument("--output", type=Path, required=True)
    quant.add_argument("--execute", action="store_true")

    calibrate = commands.add_parser("quant-calibrate", help="校準已準備的 Q1 模擬")
    calibrate.add_argument("--checkpoint", type=Path, required=True)
    calibrate.add_argument("--calibration-tensors", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--batch-size", type=int, default=16)
    calibrate.add_argument("--device", default="0")
    calibrate.add_argument("--max-batches", type=int)
    calibrate.add_argument("--execute", action="store_true")

    materialize = commands.add_parser("materialize-bittrue")
    materialize.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    materialize.add_argument("--checkpoint", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--execute", action="store_true")

    validate = commands.add_parser("validate-bittrue")
    validate.add_argument("--task", choices=("detect", "pose"), default="detect")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--enable-pose", action="store_true", help="當 task=pose 時明確同意執行")
    validate.add_argument("--execute", action="store_true")

    profile = commands.add_parser("profile")
    profile.add_argument("--checkpoint", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--warmup", type=int, default=20)
    profile.add_argument("--iterations", type=int, default=100)
    profile.add_argument("--execute", action="store_true")

    gaps = commands.add_parser("quant-report")
    gaps.add_argument("--q0", type=float, required=True)
    gaps.add_argument("--q1", type=float, required=True)
    gaps.add_argument("--q2", type=float, required=True)
    return parser


def _load_metrics(path: Path):
    from .decisions import CandidateMetrics

    return CandidateMetrics(**json.loads(path.read_text(encoding="utf-8")))


def _assert_c0_or_c_best(root: Path, candidate: str) -> None:
    if candidate == "C0":
        return
    selection = root / "artifacts/selection.json"
    if not selection.is_file():
        raise RuntimeError("尚未選出 C_best；下游工作目前只允許 C0")
    selected = json.loads(selection.read_text(encoding="utf-8")).get("c_best")
    if not isinstance(selected, dict) or selected.get("metrics", {}).get("candidate_id") != candidate:
        raise RuntimeError(f"下游工作只允許 C0 與已記錄的 C_best，不允許 {candidate}")


def _lineage_payload(root: Path, candidate: str, parent: Path, stage: str = "D1") -> dict[str, Any]:
    from .config import compose_training_config, file_sha256, manifest_hashes

    training = compose_training_config(project_root=root, task="detect", candidate_id=candidate, stage=stage)
    dataset = (root / str(training.args["data"])).resolve()
    payload: dict[str, Any] = manifest_hashes(
        spec_path=root / "EXPERIMENT_SPEC.md",
        architecture_path=CANDIDATES[candidate].config_path,
        training=training,
        dataset_path=dataset,
        parent_checkpoint=parent,
    )
    quant = root / "configs/quant/w8a8-simulation.yaml"
    payload["quantization_yaml_sha256"] = file_sha256(quant)
    return payload


def _write_lineage(destination: Path, payload: dict[str, Any]) -> Path:
    path = destination.parent / "lineage.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command in {"build", "build-pose"} and args.seed != 0:
        raise ValueError("正式候選與 Pose head 的 seed 固定為 0")
    if args.command == "build-pose" and args.execute and not args.enable_pose:
        raise ValueError("執行 Pose 建構必須同時提供 --enable-pose 與 --execute")
    if args.command == "validate-bittrue" and args.task == "pose" and args.execute and not args.enable_pose:
        raise ValueError("執行 Pose 驗證必須同時提供 --enable-pose 與 --execute")

    if args.command == "config-check":
        from .config import check_configs

        _print(check_configs(root).to_dict())
        return 0

    if args.command == "prepare-pose-data":
        from .pose_data import prepare_grouped_pose_dataset

        report = prepare_grouped_pose_dataset(
            _resolve(root, args.source),
            _resolve(root, args.destination),
            execute=args.execute,
            expected_patch_count=4,
        )
        _print(report.to_dict())
        return 0

    if args.command == "preflight":
        import importlib.util

        import torch
        import ultralytics

        from .config import check_configs

        accepted = root / "artifacts/intake/accepted.json"
        config_report = check_configs(root)
        payload = {
            "valid": bool(
                torch.cuda.is_available()
                and importlib.util.find_spec("pycocotools") is not None
                and accepted.is_file()
                and ultralytics.__version__ == "8.4.90"
                and config_report.valid
            ),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pycocotools_installed": importlib.util.find_spec("pycocotools") is not None,
            "accepted_handoff": accepted.is_file(),
            "config_check": config_report.to_dict(),
        }
        _print(payload)
        return 0 if payload["valid"] else 1

    if args.command == "intake":
        from .intake import HandoffManifest, validate_handoff, write_intake

        manifest = HandoffManifest.load(_resolve(root, args.manifest))
        payload: dict[str, Any] = {
            "manifest": str(manifest.source_manifest),
            "variant": manifest.variant,
            "float_checkpoint": str(manifest.float_checkpoint.path),
            "bittrue_checkpoint": str(manifest.bittrue_checkpoint.path),
            "will_execute": args.execute,
        }
        if args.execute:
            report = validate_handoff(manifest.source_manifest, project_root=root)
            destination = write_intake(report, root / "artifacts/intake/accepted.json")
            payload.update(status="accepted", report=str(destination), validation=asdict(report))
        _print(payload)
        return 0

    if args.command == "inspect":
        from ultralytics import YOLO

        from .graph import inspect_graph

        checkpoint = _resolve(root, args.checkpoint).resolve()
        report = inspect_graph(YOLO(str(checkpoint)).model, require_masf=not args.allow_missing_masf)
        _print(report.to_dict())
        return 0

    if args.command == "build":
        from .intake import require_accepted_intake

        intake = require_accepted_intake(root)
        output = _resolve(
            root,
            args.output or Path(f"artifacts/candidates/{args.candidate.lower()}/float-parent.pt"),
        ).resolve()
        payload = {
            "candidate": args.candidate,
            "same_float_parent": intake["float_checkpoint"],
            "output": str(output),
            "seed": args.seed,
            "will_execute": args.execute,
        }
        if args.execute:
            from ultralytics import YOLO

            from .candidate import build_candidate, write_build_report
            from .config import compose_training_config, manifest_hashes
            from .graph import write_graph_snapshot
            from .intake import file_sha256

            source = Path(intake["float_checkpoint"]["path"])
            yolo = YOLO(str(source))
            model, report = build_candidate(yolo.model, args.candidate, seed=args.seed)
            yolo.model = model
            output.parent.mkdir(parents=True, exist_ok=False)
            yolo.save(str(output))
            report_path = write_build_report(report, output.parent / "transfer-report.json")
            snapshot_path = write_graph_snapshot(
                model, args.candidate, output.parent / "architecture-snapshot.yaml"
            )
            training = compose_training_config(
                project_root=root, task="detect", candidate_id=args.candidate, stage="D1"
            )
            dataset_path = root / str(training.args["data"])
            hashes = manifest_hashes(
                spec_path=root / "EXPERIMENT_SPEC.md",
                architecture_path=CANDIDATES[args.candidate].config_path,
                training=training,
                dataset_path=dataset_path,
                parent_checkpoint=source,
            )
            lineage = {
                "schema_version": 2,
                "candidate_id": args.candidate,
                "lineage": hashes,
                "architecture_yaml": str(CANDIDATES[args.candidate].config_path),
                "formal_training_yaml": str(training.sources[0]),
                "dataset_yaml": str(dataset_path.resolve()),
                "parent": {"path": str(source), "sha256": file_sha256(source)},
                "checkpoint": {"path": str(output), "sha256": file_sha256(output)},
                "transfer_report": str(report_path),
                "architecture_snapshot": str(snapshot_path),
            }
            (output.parent / "lineage.json").write_text(
                json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            payload.update(status="built", report=report.to_dict(), checkpoint_sha256=file_sha256(output))
        _print(payload)
        return 0

    if args.command == "build-pose":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        _assert_c0_or_c_best(root, args.candidate)
        checkpoint = _resolve(root, args.checkpoint).resolve()
        output = _resolve(
            root,
            args.output or Path(f"artifacts/pose/candidates/{args.candidate.lower()}/pose-graft.pt"),
        ).resolve()
        payload = {
            "candidate": args.candidate,
            "detect_checkpoint": str(checkpoint),
            "output": str(output),
            "head": "official-local-Pose26",
            "head_seed": args.seed,
            "will_execute": args.execute,
            "pose_opt_in_required": True,
            "pose_enabled": args.enable_pose,
        }
        if args.execute:
            from ultralytics import YOLO

            from .candidate import graft_pose_candidate, write_build_report
            from .config import compose_training_config, manifest_hashes
            from .graph import inspect_graph, write_graph_snapshot
            from .intake import file_sha256

            data_yaml = root / "configs/data/pose-grouped.yaml"
            yolo = YOLO(str(checkpoint))
            pose_model, report = graft_pose_candidate(
                yolo.model,
                args.candidate,
                data_yaml=data_yaml,
                seed=args.seed,
            )
            yolo.model = pose_model
            yolo.task = "pose"
            output.parent.mkdir(parents=True, exist_ok=False)
            yolo.save(str(output))
            reloaded = YOLO(str(output))
            reload_graph = inspect_graph(reloaded.model)
            if reload_graph.task != "pose" or reload_graph.head_type != "Pose26":
                raise AssertionError("Pose26 checkpoint 無法在新程序重新載入")
            report_path = write_build_report(report, output.parent / "transfer-report.json")
            snapshot_path = write_graph_snapshot(
                pose_model, args.candidate, output.parent / "architecture-snapshot.yaml"
            )
            overlay_id = "C0" if args.candidate == "C0" else "C_best"
            training = compose_training_config(
                project_root=root, task="pose", candidate_id=overlay_id, stage="P1"
            )
            hashes = manifest_hashes(
                spec_path=root / "EXPERIMENT_SPEC.md",
                architecture_path=CANDIDATES[args.candidate].config_path,
                training=training,
                dataset_path=data_yaml,
                parent_checkpoint=checkpoint,
            )
            lineage = {
                "schema_version": 2,
                "candidate_id": args.candidate,
                "task": "pose",
                "head": "Pose26",
                "head_seed": args.seed,
                "lineage": hashes,
                "architecture_yaml": str(CANDIDATES[args.candidate].config_path),
                "formal_training_yaml": str(training.sources[0]),
                "dataset_yaml": str(data_yaml),
                "parent": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
                "checkpoint": {"path": str(output), "sha256": file_sha256(output)},
                "transfer_report": str(report_path),
                "architecture_snapshot": str(snapshot_path),
                "fresh_reload": True,
            }
            (output.parent / "lineage.json").write_text(
                json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            payload.update(status="built-and-reloaded", report=report.to_dict(), lineage=lineage)
        _print(payload)
        return 0

    if args.command == "train":
        from .config import compose_training_config, load_formal_training_config
        from .training import STAGE_RULES, normalize_stage

        checkpoint = _resolve(root, args.checkpoint).resolve()
        config_path: Path | None = None
        if args.config is not None:
            if args.stage is not None:
                raise ValueError("使用 --config 時不可再提供 --stage；task/stage 已由正式 YAML 定義")
            config_path = _resolve(root, args.config).resolve()
            formal = load_formal_training_config(
                config_path,
                candidate_id=args.candidate,
                project_root=root,
            )
            if args.task is not None and args.task != formal.task:
                raise ValueError(
                    f"--task={args.task} 與正式設定 task={formal.task} 不一致"
                )
        else:
            if args.stage is None:
                raise ValueError("請提供 --config；相容模式則必須提供 --stage")
            task = args.task or "detect"
            normalized = normalize_stage(args.stage, task)
            formal = compose_training_config(
                project_root=root,
                task=task,
                candidate_id=args.candidate,
                stage=normalized,
            )
            config_path = formal.sources[0]
        task = formal.task
        normalized_stage = formal.stage
        if task == "pose" and args.execute and not args.enable_pose:
            raise ValueError("執行 Pose 訓練必須同時提供 --enable-pose 與 --execute")
        payload = {
            "config": str(config_path),
            "config_id": formal.config_id,
            "title_zh": formal.title_zh,
            "candidate": args.candidate,
            "checkpoint": str(checkpoint),
            "task": task,
            "stage": normalized_stage,
            "stage_rules": STAGE_RULES[normalized_stage],
            "run_id": args.run_id,
            "will_execute": args.execute,
            "pose_opt_in_required": task == "pose",
            "pose_enabled": args.enable_pose,
            "ultralytics_yaml_direct": False,
        }
        if task == "pose" or normalized_stage == "Q2":
            _assert_c0_or_c_best(root, args.candidate)
        if args.execute:
            from .training import launch_training

            payload["completion"] = str(
                launch_training(
                    project_root=root,
                    checkpoint=checkpoint,
                    candidate_id=args.candidate,
                    stage=normalized_stage,
                    run_id=args.run_id,
                    task=task,
                    smoke_epochs=args.smoke_epochs,
                    pose_opt_in=args.enable_pose,
                    training_config_path=config_path,
                )
            )
        _print(payload)
        return 0

    if args.command == "extension-gate":
        from .decisions import should_extend

        metrics = json.loads(_resolve(root, args.metrics).read_text(encoding="utf-8"))
        _print(asdict(should_extend(metrics, best_epoch=args.best_epoch, early_stopped=args.early_stopped)))
        return 0

    if args.command == "assess":
        from .decisions import (
            choose_c_best,
            classify_candidate,
            trigger_c3_p5_fallback,
            trigger_r1,
            validate_conditional_candidates,
        )

        c0 = _load_metrics(_resolve(root, args.c0))
        decisions = [classify_candidate(_load_metrics(_resolve(root, path)), c0) for path in args.candidate]
        fusion_passed = False
        fusion_report = None
        if args.r1_fusion_report:
            fusion_report = json.loads(_resolve(root, args.r1_fusion_report).read_text(encoding="utf-8"))
            fusion_passed = bool(
                fusion_report.get("candidate_id") == "R1"
                and fusion_report.get("passed")
                and float(fusion_report.get("max_abs_diff", float("inf"))) <= 1e-4
            )
        validate_conditional_candidates(decisions, r1_fusion_passed=fusion_passed)
        winner = choose_c_best(decisions)
        payload = {
            "c0": asdict(c0),
            "decisions": [asdict(item) for item in decisions],
            "triggers": {
                "c3_p5": any(trigger_c3_p5_fallback(item) for item in decisions),
                "r1": any(trigger_r1(item) for item in decisions),
            },
            "c_best": asdict(winner) if winner else None,
            "r1_fusion_report": fusion_report,
            "quantization_mainline_gated": winner is None,
        }
        if args.execute:
            destination = root / "artifacts/selection.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            payload["selection_artifact"] = str(destination)
        _print(payload)
        return 0

    if args.command == "rep-fusion":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        checkpoint = _resolve(root, args.checkpoint).resolve()
        output = _resolve(root, args.output).resolve()
        payload = {
            "candidate_id": "R1",
            "checkpoint": str(checkpoint),
            "output": str(output),
            "tolerance": 1e-4,
            "will_execute": args.execute,
        }
        if args.execute:
            import torch
            from ultralytics import YOLO

            from .config import SPEC_VERSION, file_sha256
            from .graph import assert_candidate_graph
            from .rep import assert_rep_fuse

            model = YOLO(str(checkpoint)).model.eval()
            assert_candidate_graph(model, "R1", CANDIDATES["R1"].target_layers)
            parameter = next(model.parameters())
            sample = torch.zeros(
                1,
                3,
                640,
                640,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            report = assert_rep_fuse(model, sample, tolerance=1e-4)
            result = {
                "schema_version": 1,
                "spec_version": SPEC_VERSION,
                "spec_sha256": file_sha256(root / "EXPERIMENT_SPEC.md"),
                "candidate_id": "R1",
                "checkpoint_sha256": file_sha256(checkpoint),
                **asdict(report),
            }
            if output.exists():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            payload.update(result)
        _print(payload)
        return 0

    if args.command == "fuse-reference":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        _assert_c0_or_c_best(root, args.candidate)
        output = _resolve(root, args.output).resolve()
        payload = {
            "candidate": args.candidate,
            "output": str(output),
            "will_execute": args.execute,
        }
        if args.execute:
            from ultralytics import YOLO

            from .graph import inspect_graph
            from .intake import file_sha256
            from .quantization import make_fused_reference

            yolo = YOLO(str(_resolve(root, args.checkpoint).resolve()))
            yolo.model = make_fused_reference(yolo.model)
            inspect_graph(yolo.model)
            output.parent.mkdir(parents=True, exist_ok=False)
            yolo.save(str(output))
            inspect_graph(YOLO(str(output)).model)
            lineage = _lineage_payload(root, args.candidate, _resolve(root, args.checkpoint).resolve())
            lineage.update(
                candidate_id=args.candidate,
                checkpoint={"path": str(output), "sha256": file_sha256(output)},
                stage="Q0",
                simulation_only=True,
            )
            payload.update(
                status="fused-and-reloaded",
                lineage=str(_write_lineage(output, lineage)),
            )
        _print(payload)
        return 0

    if args.command == "quant-prepare":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        _assert_c0_or_c_best(root, args.candidate)
        output = _resolve(root, args.output).resolve()
        payload = {
            "candidate": args.candidate,
            "checkpoint": str(_resolve(root, args.checkpoint).resolve()),
            "output": str(output),
            "simulation_only": True,
            "will_execute": args.execute,
        }
        if args.execute:
            import torch
            from ultralytics import YOLO

            from .quantization import prepare_w8a8_simulation, quant_scope_dict

            yolo = YOLO(str(_resolve(root, args.checkpoint).resolve()))
            prepared, scope = prepare_w8a8_simulation(yolo.model)
            lineage = _lineage_payload(root, args.candidate, _resolve(root, args.checkpoint).resolve())
            lineage.update(candidate_id=args.candidate, stage="Q1-or-Q2-prepare", simulation_only=True)
            output.parent.mkdir(parents=True, exist_ok=False)
            torch.save(
                {
                    "model": prepared,
                    "simulation_only": True,
                    "candidate_id": args.candidate,
                    "lineage": lineage,
                },
                output,
            )
            scope_path = output.parent / "quant-scope.json"
            scope_path.write_text(
                json.dumps(quant_scope_dict(scope), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            lineage_path = _write_lineage(output, lineage)
            payload.update(
                scope=quant_scope_dict(scope),
                scope_artifact=str(scope_path),
                lineage=str(lineage_path),
            )
        _print(payload)
        return 0

    if args.command == "quant-calibrate":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        output = _resolve(root, args.output).resolve()
        payload = {"output": str(output), "simulation_only": True, "will_execute": args.execute}
        if args.execute:
            import torch

            from .quantization import calibrate_w8a8

            checkpoint = torch.load(_resolve(root, args.checkpoint), map_location="cpu", weights_only=False)
            model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
            if not isinstance(model, torch.nn.Module):
                raise TypeError("quant calibration checkpoint must contain model")
            images = torch.load(
                _resolve(root, args.calibration_tensors), map_location="cpu", weights_only=True
            )
            if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1:] != (3, 640, 640):
                raise ValueError("校準張量形狀必須為 [N, 3, 640, 640]")
            count = calibrate_w8a8(model, images.split(args.batch_size), max_batches=args.max_batches)
            output.parent.mkdir(parents=True, exist_ok=False)
            lineage = checkpoint.get("lineage") if isinstance(checkpoint, dict) else None
            if isinstance(lineage, dict):
                lineage = {**lineage, "stage": "Q1-calibrated", "calibration_batches": count}
            torch.save(
                {**checkpoint, "model": model, "calibration_batches": count, "lineage": lineage},
                output,
            )
            payload["calibration_batches"] = count
            if isinstance(lineage, dict):
                payload["lineage"] = str(_write_lineage(output, lineage))
        _print(payload)
        return 0

    if args.command == "materialize-bittrue":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        _assert_c0_or_c_best(root, args.candidate)
        output = _resolve(root, args.output).resolve()
        payload = {"candidate": args.candidate, "output": str(output), "will_execute": args.execute}
        if args.execute:
            from achitechure_1.checkpoint import materialize_bittrue_checkpoint

            from .intake import file_sha256

            parent = _resolve(root, args.checkpoint).resolve()
            payload["output"] = str(
                materialize_bittrue_checkpoint(
                    parent,
                    root.parent / "achitechure_1/configs/attention/bittrue-pwl-final.yaml",
                    output,
                )
            )
            lineage = _lineage_payload(root, args.candidate, parent)
            lineage.update(
                candidate_id=args.candidate,
                stage="bittrue-materialization",
                checkpoint={"path": str(output), "sha256": file_sha256(output)},
            )
            payload["lineage"] = str(_write_lineage(output, lineage))
        _print(payload)
        return 0

    if args.command == "validate-bittrue":
        from .config import compose_training_config
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        stage = "D1" if args.task == "detect" else "P3"
        config = compose_training_config(
            project_root=root,
            task=args.task,
            candidate_id="C0",
            stage=stage,
        )
        common = config.args
        run = root / "artifacts/validation" / args.run_id
        payload = {
            "task": args.task,
            "run": str(run),
            "will_execute": args.execute,
            "pose_opt_in_required": args.task == "pose",
            "pose_enabled": args.enable_pose,
        }
        if args.execute and args.task == "detect":
            from achitechure_1.evaluation import validate_bittrue

            payload["metrics"] = str(
                validate_bittrue(
                    checkpoint=_resolve(root, args.checkpoint),
                    data=_resolve(root, Path(common["data"])),
                    run_dir=run,
                    imgsz=int(common["imgsz"]),
                    batch=int(common["batch"]),
                    device=str(common["device"]),
                    workers=int(common["workers"]),
                )
            )
        elif args.execute:
            from ultralytics import YOLO

            result = YOLO(str(_resolve(root, args.checkpoint))).val(
                data=str(_resolve(root, Path(common["data"]))),
                imgsz=int(common["imgsz"]),
                batch=int(common["batch"]),
                device=str(common["device"]),
                workers=int(common["workers"]),
                project=str(run),
                name="ultralytics",
                exist_ok=False,
            )
            metrics = dict(getattr(result, "results_dict", {}))
            run.mkdir(parents=True, exist_ok=True)
            destination = run / "metrics.json"
            destination.write_text(
                json.dumps(metrics, indent=2, sort_keys=True, default=float) + "\n",
                encoding="utf-8",
            )
            payload["metrics"] = str(destination)
        _print(payload)
        return 0

    if args.command == "profile":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        payload = {"output": str(_resolve(root, args.output)), "will_execute": args.execute}
        if args.execute:
            from achitechure_1.profiling import profile_checkpoint

            payload["output"] = str(
                profile_checkpoint(
                    checkpoint=_resolve(root, args.checkpoint),
                    output=_resolve(root, args.output),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )
        _print(payload)
        return 0

    if args.command == "quant-report":
        from .quantization import robustness_report

        _print(asdict(robustness_report(args.q0, args.q1, args.q2)))
        return 0
    raise AssertionError("未處理的命令")


if __name__ == "__main__":
    raise SystemExit(main())
