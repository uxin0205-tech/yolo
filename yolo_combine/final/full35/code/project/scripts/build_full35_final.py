#!/usr/bin/env python3
"""建立可稽核、可推論、可驗證與可續訓的 Full35 final 交付包。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPO / "final" / "full35"
PARENT_RUN = (
    REPO
    / "variants/full35/artifacts/fusion/formal/"
    "full35-joint-adamw-v2-j0e8-j1e20-j2e80-seed0"
)
ACCEPTED_RUN = (
    REPO
    / "variants/full35/artifacts/fusion/formal/"
    "full35-joint-adamw-v2-j3-b32-challenger-seed0"
)
PARENT_EVALUATION = (
    REPO
    / "variants/full35/artifacts/fusion/formal/evaluations/"
    "full35-v2-best-joint-seed0/epoch-0000"
)
FINAL_EVALUATION = (
    REPO
    / "variants/full35/artifacts/fusion/formal/evaluations/"
    "full35-v2-j3-best-joint-seed0/epoch-0000"
)
FINAL_REPORT = REPO / "reports/full35"
SOURCE_BUNDLE = Path(
    "/home/uxin/yolo/yolo_achitechure/achitechure_1/final"
)
POSE_P3 = (
    REPO
    / "variants/full35/artifacts/pose/"
    "p0-full35-p3-b32a4-e100max-seed0/weights/best.pt"
)
BASELINE = REPO / "variants/full35/baselines/formal-gate-p3-seed0.json"
INDEPENDENT_BASELINE = REPO / "variants/full35/baselines/independent.json"
JOINT_CONFIG = REPO / "variants/full35/configs/joint.yaml"
J2_QUEUE = REPO / "variants/full35/artifacts/queue/full35-v2-seed0/queue-status.json"
J3_QUEUE = (
    REPO
    / "variants/full35/artifacts/queue/"
    "full35-v2-j3-b32-seed0/queue-status.json"
)
PERSON_AP_WORKLOG = REPO / "docs/worklogs/2026-08-27-coco-person-ap-selection.md"

COMBINED_FILES = (
    "best_detect.pt",
    "best_pose.pt",
    "best_joint.pt",
    "last.pt",
)
GATE_METRICS = (
    "coco/box/map50_95",
    "coco/person/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
    "bbat/ball/box/map50_95",
    "bbat/bat/box/map50_95",
    "bbat/ball/pose/map50_95",
    "bbat/bat/pose/map50_95",
)
MANIFEST_EXCLUSIONS = frozenset({"MANIFEST.json", "CHECKSUMS.sha256"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_python_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        _copy(path, destination / path.relative_to(source))


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        destination,
        copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.cache"),
    )


def _regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"final package must not contain symlinks: {path}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_file() and path.name not in MANIFEST_EXCLUSIONS:
            yield path


def _manifest(root: Path) -> dict[str, Any]:
    records = []
    total = 0
    for path in _regular_files(root):
        size = path.stat().st_size
        total += size
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "files": records,
        "file_count": len(records),
        "total_bytes": total,
    }


def _write_manifest(root: Path) -> dict[str, Any]:
    payload = _manifest(root)
    _write_json(root / "MANIFEST.json", payload)
    lines = [f"{item['sha256']}  {item['path']}" for item in payload["files"]]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def verify(root: Path) -> dict[str, Any]:
    manifest_path = root / "MANIFEST.json"
    payload = _json(manifest_path)
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("MANIFEST.json contains no file records")
    expected: set[str] = set()
    verified_bytes = 0
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("manifest record must be a mapping")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        name = relative.as_posix()
        if name in expected:
            raise ValueError(f"duplicate manifest path: {name}")
        expected.add(name)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(record["bytes"]):
            raise ValueError(f"size mismatch: {name}")
        if _sha256(path) != str(record["sha256"]):
            raise ValueError(f"SHA256 mismatch: {name}")
        verified_bytes += size
    actual = {path.relative_to(root).as_posix() for path in _regular_files(root)}
    if actual != expected:
        raise ValueError(
            "manifest coverage mismatch; "
            f"missing={sorted(actual - expected)}, extra={sorted(expected - actual)}"
        )
    checksum_lines = (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    if len(checksum_lines) != len(records):
        raise ValueError("CHECKSUMS.sha256 line count differs from manifest")
    return {
        "valid": True,
        "root": str(root.resolve()),
        "files": len(records),
        "bytes": verified_bytes,
        "manifest_sha256": _sha256(manifest_path),
    }


def _copy_project_code(root: Path) -> None:
    project = root / "code/project"
    _copy_python_tree(REPO / "src", project / "src")
    _copy_python_tree(REPO / "tests", project / "tests")
    _copy_python_tree(REPO / "scripts", project / "scripts")
    _copy(REPO / "pyproject.toml", project / "pyproject.toml")
    for name in ("AGENTS.md", "CONTEXT.md", "README.md"):
        path = REPO / name
        if path.is_file():
            _copy(path, project / name)
    _copy_tree(REPO / "docs", project / "docs")
    variant = project / "variants/full35"
    for name in ("joint.py", "baseline.py", "cpu_joint_smoke.py", "run.py", "README.md"):
        path = REPO / "variants/full35" / name
        if path.is_file():
            _copy(path, variant / name)
    _copy_tree(REPO / "variants/full35/configs", variant / "configs")


def _copy_source_bundle(root: Path) -> None:
    bundle = root / "source_bundle"
    _copy_python_tree(SOURCE_BUNDLE / "code/achitechure_1", bundle / "code/achitechure_1")
    _copy_python_tree(SOURCE_BUNDLE / "code/yolo_attention", bundle / "code/yolo_attention")
    for name in ("float-pwl-final.yaml", "bittrue-pwl-final.yaml"):
        _copy(
            SOURCE_BUNDLE / "configs/attention" / name,
            bundle / "configs/attention" / name,
        )
    for kind in ("float", "bittrue"):
        _copy(
            SOURCE_BUNDLE / "weights" / kind / "full35-a2.pt",
            bundle / "weights" / kind / "full35-a2.pt",
        )
    for name in ("SOURCE_PROVENANCE.md", "VALIDATION.md", "requirements-lock.txt"):
        path = SOURCE_BUNDLE / name
        if path.is_file():
            _copy(path, bundle / name)
    _copy(SOURCE_BUNDLE / "README.md", bundle / "ORIGINAL_BUNDLE_README.md")
    (bundle / "README.md").write_text(
        "# Full35 final 專用來源子集\n\n"
        "此處不是原始 9 候選 Full35／Partial75 研究 bundle 的完整複本，而是供 "
        "`final/full35` 重建 graph 所需的受控子集。\n\n"
        "保留內容：\n\n"
        "- `code/achitechure_1/` 與 `code/yolo_attention/` 的完整 Python 模組；\n"
        "- Full35-A2 Float／Bit-True checkpoint 各一份；\n"
        "- Float／Bit-True attention YAML 與精確 dependency lock；\n"
        "- 原始來源與驗證說明。\n\n"
        "刻意排除 Partial75、A0、B/C ablation、報表與其他淘汰權重。"
        "原始 bundle 說明另存 `ORIGINAL_BUNDLE_README.md`，只作血緣證據；"
        "本子集的實際內容以此目錄 `MANIFEST.json` 為準。\n",
        encoding="utf-8",
    )
    source_manifest = _manifest(bundle)
    _write_json(bundle / "MANIFEST.json", source_manifest)


def _copy_weights(root: Path) -> None:
    primary_sources = {
        "best_detect.pt": PARENT_RUN,
        "best_pose.pt": ACCEPTED_RUN,
        "best_joint.pt": ACCEPTED_RUN,
        "last.pt": ACCEPTED_RUN,
    }
    kinds = {"checkpoints": "full-resume", "inference": "inference"}
    for name, source_run in primary_sources.items():
        for source_kind, destination_kind in kinds.items():
            _copy(
                source_run / source_kind / name,
                root / "weights/combined" / destination_kind / name,
            )
    for name in COMBINED_FILES:
        for source_kind, destination_kind in kinds.items():
            _copy(
                PARENT_RUN / source_kind / name,
                root / "weights/rollback/j2" / destination_kind / name,
            )
    (root / "weights/rollback/j2/README.md").write_text(
        "# J2 rollback\n\n"
        "這裡保留升格J3前的完整J2四種selector權重。正式主權重已是J3；"
        "只有`weights/combined/*/best_detect.pt`仍沿用J2，因J3沒有刷新Detect selector。\n",
        encoding="utf-8",
    )
    _copy(
        SOURCE_BUNDLE / "weights/float/full35-a2.pt",
        root / "weights/standalone/detect-full35-a2-float.pt",
    )
    _copy(
        SOURCE_BUNDLE / "weights/bittrue/full35-a2.pt",
        root / "weights/standalone/detect-full35-a2-bittrue.pt",
    )
    _copy(POSE_P3, root / "weights/standalone/pose-full35-p3.pt")


def _copy_outputs(root: Path) -> None:
    _copy_tree(ACCEPTED_RUN / "logs", root / "outputs/training/logs")
    for name in ("resolved-config.json", "factory-report.json"):
        _copy(ACCEPTED_RUN / name, root / "outputs/training" / name)
    _copy_tree(PARENT_RUN / "logs", root / "outputs/training/parent-j2/logs")
    for name in ("resolved-config.json", "factory-report.json"):
        _copy(PARENT_RUN / name, root / "outputs/training/parent-j2" / name)
    _copy_tree(FINAL_EVALUATION, root / "outputs/validation/accepted-best-joint")
    _copy_tree(
        PARENT_EVALUATION,
        root / "outputs/validation/parent-j2-best-joint",
    )
    _copy_tree(FINAL_REPORT, root / "analysis")


def _copy_configs_and_provenance(root: Path) -> None:
    _copy(JOINT_CONFIG, root / "configs/experiment-joint.yaml")
    _copy(BASELINE, root / "metrics/standalone-baseline-bittrue.json")
    _copy(INDEPENDENT_BASELINE, root / "metrics/independent-history.json")
    external = {
        "bbat5_registry.yaml": Path("/home/uxin/yolo/configs/datasets/bbat5-v1.yaml"),
        "bbat5_pose.yaml": Path(
            "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml"
        ),
        "bbat5_detect.yaml": Path(
            "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml"
        ),
        "coco2017.yaml": Path("/home/uxin/yolo/coco2017.yaml"),
    }
    for name, source in external.items():
        _copy(source, root / "configs/data-snapshots" / name)
    for name, source in (("j2-queue-status.json", J2_QUEUE), ("j3-queue-status.json", J3_QUEUE)):
        if source.is_file():
            _copy(source, root / "provenance" / name)


def _final_joint_config(root: Path) -> None:
    payload = yaml.safe_load(JOINT_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("joint config must be a mapping")
    payload["source"]["bundle"] = "../source_bundle"
    payload["source"]["pose_checkpoint"] = "../weights/standalone/pose-full35-p3.pt"
    payload["source"]["provisional_pose_checkpoint"] = None
    payload["source"]["baseline_metrics"] = "../metrics/standalone-baseline-bittrue.json"
    payload["runs"]["root"] = "../runtime"
    target = root / "configs/joint.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _gate_report(root: Path) -> dict[str, Any]:
    baseline = _json(BASELINE)["metrics"]
    evaluation = _json(FINAL_EVALUATION / "bittrue/metrics.json")["metrics"]
    rows = []
    for metric in GATE_METRICS:
        base = float(baseline[metric])
        actual = float(evaluation[metric])
        rows.append(
            {
                "metric": metric,
                "standalone": base,
                "combined": actual,
                "delta": actual - base,
                "passed": actual - base >= -0.08,
            }
        )
    target = root / "metrics/gate-deltas.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "threshold": -0.08,
        "passed": all(row["passed"] for row in rows),
        "metrics": rows,
    }


def _release_status(root: Path, gate: dict[str, Any]) -> dict[str, Any]:
    best = _json(FINAL_EVALUATION / "bittrue/metrics.json")["metrics"]
    float_metrics = _json(FINAL_EVALUATION / "float/metrics.json")["metrics"]
    factory = _json(ACCEPTED_RUN / "factory-report.json")
    j3 = _json(J3_QUEUE) if J3_QUEUE.is_file() else {"status": "not_found"}
    analysis = _json(FINAL_REPORT / "SUMMARY.json")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "YOLO26m Full35 shared trunk + Detect80 + Pose26(ball, bat)",
        "release_state": "accepted_j3_with_j2_rollback",
        "accepted_candidate": {
            "stage": "j3",
            "run": ACCEPTED_RUN.name,
            "best_global_epoch": 58,
            "best_j3_local_epoch_zero_based": 5,
            "joint_score": float(analysis["accepted"]["joint_score"]),
            "selection_backend": "bittrue",
            "inference_weight": "weights/combined/inference/best_joint.pt",
            "full_resume_weight": "weights/combined/full-resume/best_joint.pt",
            "resume_last_weight": "weights/combined/full-resume/last.pt",
            "best_detect_source": "parent_j2",
        },
        "deployment_selection": analysis["deployment_selection"],
        "promotion_record": {
            "queue_status": j3.get("status", "unknown"),
            "candidate_improved_parent_best": j3.get(
                "candidate_improved_parent_best", False
            ),
            "parent_joint_score": float(analysis["parent_j2"]["joint_score"]),
            "improvement_over_parent": float(analysis["j3_improvement_over_j2"]),
            "decision": "promoted_after_all_eight_gates_passed",
        },
        "rollback": {
            "stage": "j2",
            "run": PARENT_RUN.name,
            "joint_score": float(analysis["parent_j2"]["joint_score"]),
            "full_resume": "weights/rollback/j2/full-resume/best_joint.pt",
            "inference": "weights/rollback/j2/inference/best_joint.pt",
        },
        "analysis": {
            "report": "analysis/FINAL_ANALYSIS.md",
            "summary": "analysis/SUMMARY.json",
            "figures": analysis["figures"],
        },
        "model_contract": {
            "shared_layers": factory["assembly"]["shared_layers"],
            "head_inputs": factory["assembly"]["head_inputs"],
            "feature_channels": factory["assembly"]["feature_channels"],
            "strides": factory["assembly"]["strides"],
            "detect_nc": factory["assembly"]["detect_nc"],
            "pose_nc": factory["assembly"]["pose_nc"],
            "pose_kpt_shape": factory["assembly"]["pose_kpt_shape"],
            "independent_parameters": factory["assembly"]["independent_parameters"],
            "shared_parameters": factory["assembly"]["shared_parameters"],
            "parameter_reduction_fraction": factory["assembly"]["parameter_reduction_fraction"],
            "xnor": factory["xnor"],
        },
        "bittrue_key_metrics": {name: float(best[name]) for name in GATE_METRICS},
        "float_key_metrics": {name: float(float_metrics[name]) for name in GATE_METRICS},
        "hard_gate": gate,
        "data_policy": {
            "detect": "/home/uxin/yolo/coco2017.yaml",
            "pose": "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml",
            "bbat5_id": "bbat5-v1",
            "dataset_payload_copied": False,
            "reason": "資料本體維持 canonical 唯一來源；final 僅保存 YAML/registry snapshot。",
        },
        "dependencies": {
            "python": ">=3.12,<3.13",
            "torch": "2.11.0+cu128",
            "ultralytics": "8.4.90",
            "compile": False,
            "ddp": False,
        },
    }


def _write_entrypoints(root: Path) -> None:
    run_py = '''#!/usr/bin/env python3
"""Final Full35 CLI: preflight, train/resume, validate and infer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code/project/src"))

from yolo_combine.joint_cli import main

if __name__ == "__main__":
    main(ROOT / "configs/joint.yaml")
'''
    verify_py = '''#!/usr/bin/env python3
"""Verify every regular file in this final package against MANIFEST.json."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"MANIFEST.json", "CHECKSUMS.sha256"}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
expected = set()
total = 0
for record in payload["files"]:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe path: {relative}")
    path = ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"integrity mismatch: {relative}")
    expected.add(relative.as_posix())
    total += path.stat().st_size
actual = {
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file()
    and not path.is_symlink()
    and path.name not in EXCLUDED
    and "__pycache__" not in path.parts
    and path.suffix != ".pyc"
}
if actual != expected:
    raise ValueError(f"manifest coverage mismatch: missing={actual-expected}, extra={expected-actual}")
print(json.dumps({"valid": True, "files": len(expected), "bytes": total}, indent=2))
'''
    (root / "run.py").write_text(run_py, encoding="utf-8")
    (root / "verify.py").write_text(verify_py, encoding="utf-8")


def _write_readmes(root: Path, status: dict[str, Any]) -> None:
    person_candidates = {
        row["candidate_id"]: row
        for row in status["deployment_selection"]["candidates"]
    }
    person_lines = [
        "| 候選 | person AP50-95 | AP50 | AP75 | precision | recall | 八項gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for candidate_id in (
        "standalone_detect",
        "shared_best_detect",
        "j2_best_joint",
        "j3_best_joint",
        "j3_best_pose",
        "j3_last",
    ):
        row = person_candidates[candidate_id]
        gate = (
            "不適用"
            if row["all_eight_gates_passed"] is None
            else ("通過" if row["all_eight_gates_passed"] else "未通過")
        )
        person_lines.append(
            f"| {row['label']} | {row['person_ap50_95']:.6f} | "
            f"{row['person_ap50']:.6f} | {row['person_ap75']:.6f} | "
            f"{row['person_precision']:.6f} | {row['person_recall']:.6f} | {gate} |"
        )
    person_table = "\n".join(person_lines)
    readme = f"""# Full35 final 交付包

依固定實驗selector，目前預設是已通過八項 hard gate、且joint score超越J2的
**J3 best_joint**（Bit-True `{status['accepted_candidate']['joint_score']:.12f}`）。
實際部署權重尚待使用者看完COCO person AP後選定；所有候選與J2 rollback均保留，
本次沒有因報告更新而更換任何權重。

## 最直接的檔案

- 部署／推論：`weights/combined/inference/best_joint.pt`
- 從最佳點續訓：`weights/combined/full-resume/best_joint.pt`
- J3 結束點／exact state：`weights/combined/full-resume/last.pt`
- 完整程式碼快照：`code/project/`
- Full35／XNOR 必要來源：`source_bundle/`
- Float 與 Bit-True 正式 AP：`outputs/validation/accepted-best-joint/`
- 完整訓練 CSV／JSONL／圖：`outputs/training/logs/`
- 機器可讀狀態：`RELEASE_STATUS.json`
- 完整中文結論、person AP候選CSV與八張圖：`analysis/FINAL_ANALYSIS.md`
- 全檔案 SHA256：`MANIFEST.json`、`CHECKSUMS.sha256`

## 環境

必須使用 Python 3.12、PyTorch 2.11.0+cu128、Ultralytics 8.4.90；不要自行升級。`source_bundle` 是從原 721MB 研究 bundle 裁出的 Full35 必要子集，保留 Float／Bit-True A2 權重與完整 Python 模組，不含 Partial75 或其他淘汰候選。

## 驗證完整性

```bash
python verify.py
```

## 推論

```bash
python run.py infer \\
  --checkpoint weights/combined/inference/best_joint.pt \\
  --source /path/to/image.jpg \\
  --task both --device 0 \\
  --output-json outputs/inference.json
```

`--task` 可用 `detect`、`pose`、`both`。`both` 只抽取一次 shared backbone/neck feature，再分別執行 Detect 與 Pose26 head。

## 正式驗證

```bash
python run.py validate \\
  --checkpoint weights/combined/inference/best_joint.pt \\
  --backend both --device 0 --name final-recheck
```

驗證依賴本機 canonical COCO2017 與 BBAT5 v1；資料本體刻意不複製進 final。固定路徑與 YAML snapshot 在 `configs/`，正式 BBAT5 仍只能是 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`。

## 續訓與回退

J3已完成並依patience5正常停止：

- `combined/full-resume/best_joint.pt`：global epoch58的正式最佳點；
- `combined/full-resume/last.pt`：global epoch63的stage-complete終點；
- `rollback/j2/`：升格前J2四種selector的full-resume與inference版本。

`combined/*/best_detect.pt`明確沿用J2，因J3沒有刷新Detect selector。若後續跑seed1、
MuSGD或其他tuning，必須另開run；stage切換不是exact resume。

J3使用physical32×4維持logical Detect128；不可把它稱為單次physical batch128。

## 權重角色

- `combined/inference/`：只有 EMA/live state 與 model contract，不可 exact resume。
- `combined/full-resume/`：含 live/EMA、optimizer、scheduler、AMP scaler、RNG、loader、criterion progressive state。
- `rollback/j2/`：升格前J2完整回退權重。
- `standalone/`：Detect A2 與 Pose P3 回退／比較來源，不是融合推論入口。

## 目前正式結果

- shared 參數：26,529,701；兩個獨立模型合計 45,580,762；減少 41.796%。
- Bit-True COCO overall mAP50-95：{status['bittrue_key_metrics']['coco/box/map50_95']:.6f}
- Bit-True COCO person AP50-95：{status['bittrue_key_metrics']['coco/person/box/map50_95']:.6f}
- Bit-True BBAT box mAP50-95：{status['bittrue_key_metrics']['bbat/box/map50_95']:.6f}
- Bit-True BBAT pose mAP50-95：{status['bittrue_key_metrics']['bbat/pose/map50_95']:.6f}
- 八項 gate：全部通過；最差 delta 為 ball pose {min(row['delta'] for row in status['hard_gate']['metrics']):.6f}，仍高於 -0.08 下限。

## COCO person AP候選

{person_table}

`Standalone Detect`只有Detect任務；`Shared best_detect`的Pose尚未適應完成，因此不能作
`task=both`正式答案。兩個head都要時，目前預設仍是`J3 best_joint`，但`J3 best_pose`
的person AP50-95與recall略高。差距很小且目前只有seed0，最後部署選擇保留給使用者。
完整數值、joint score、epoch、provenance與兩張person專圖見`analysis/FINAL_ANALYSIS.md`。

## 邊界與風險

- 目前是 seed0 正式結果；第二 seed 尚未執行，因此跨 seed 結論仍屬 provisional。
- J3已依實驗selector升格；實際部署checkpoint仍待使用者依person AP最後決定。
- 尚未做 FPGA、HLS、真實 latency／energy 驗收。
- Ultralytics checkpoint 及衍生部署的 AGPL／商用授權需在非研究交付前另行確認。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    top = root.parent / "README.md"
    top.write_text(
        "# Final 交付區\n\n"
        "- `full35/`：J3正式Full35 shared-trunk Detect/Pose交付包。\n"
        "- `full35-j2-archive/`：升格前完整J2 package，待另行授權清理。\n"
        "- Partial75 尚未執行，不得混入 Full35；未來若啟動會另建 `partial75/`。\n",
        encoding="utf-8",
    )


def build(destination: Path) -> dict[str, Any]:
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"destination already exists; refusing to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-staging-",
        dir=destination.parent,
    ) as temporary:
        root = Path(temporary)
        _copy_project_code(root)
        _copy_source_bundle(root)
        _copy_weights(root)
        _copy_outputs(root)
        _copy_configs_and_provenance(root)
        _final_joint_config(root)
        gate = _gate_report(root)
        status = _release_status(root, gate)
        _write_json(root / "RELEASE_STATUS.json", status)
        _write_entrypoints(root)
        _write_readmes(root, status)
        manifest = _write_manifest(root)
        verified = verify(root)
        root.rename(destination)
    return {
        "built": True,
        "destination": str(destination),
        "manifest": manifest,
        "verification": verified,
        "accepted": status["accepted_candidate"],
        "promotion": status["promotion_record"],
        "rollback": status["rollback"],
    }


def refresh_report_metadata(destination: Path) -> dict[str, Any]:
    """同步報告、說明與機器狀態，不重建或更換任何checkpoint。"""

    root = destination.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(FINAL_REPORT.rglob("*")):
        if path.is_file():
            _copy(path, root / "analysis" / path.relative_to(FINAL_REPORT))
    project = root / "code/project"
    for name in ("build_full35_final.py", "build_full35_final_report.py"):
        _copy(REPO / "scripts" / name, project / "scripts" / name)
    _copy(REPO / "README.md", project / "README.md")
    _copy(REPO / "docs/worklogs/README.md", project / "docs/worklogs/README.md")
    _copy(PERSON_AP_WORKLOG, project / "docs/worklogs" / PERSON_AP_WORKLOG.name)
    gate = _gate_report(root)
    status = _release_status(root, gate)
    _write_json(root / "RELEASE_STATUS.json", status)
    _write_readmes(root, status)
    manifest = _write_manifest(root)
    return {
        "refreshed_report_metadata": True,
        "checkpoint_files_changed": False,
        "destination": str(root),
        "manifest": {
            "files": manifest["file_count"],
            "bytes": manifest["total_bytes"],
        },
        "verification": verify(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只驗證已建立的 package，不複製或修改檔案。",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="final 內容經受控更新後，重新產生 manifest 並立即驗證。",
    )
    parser.add_argument(
        "--refresh-report-metadata",
        action="store_true",
        help="只同步分析、README、狀態與manifest；不重建或更換checkpoint。",
    )
    args = parser.parse_args()
    if sum(
        bool(value)
        for value in (
            args.verify_only,
            args.refresh_manifest,
            args.refresh_report_metadata,
        )
    ) > 1:
        parser.error("--verify-only, --refresh-manifest and --refresh-report-metadata are mutually exclusive")
    if args.verify_only:
        report = verify(args.destination.resolve())
    elif args.refresh_report_metadata:
        report = refresh_report_metadata(args.destination.resolve())
    elif args.refresh_manifest:
        root = args.destination.resolve()
        _write_json(
            root / "source_bundle/MANIFEST.json",
            _manifest(root / "source_bundle"),
        )
        manifest = _write_manifest(root)
        report = {
            "refreshed": True,
            "manifest": {
                "files": manifest["file_count"],
                "bytes": manifest["total_bytes"],
            },
            "verification": verify(root),
        }
    else:
        report = build(args.destination)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
