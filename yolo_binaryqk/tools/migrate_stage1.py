"""抽取小型歷史證據，產生逐檔刪除清單；--apply 僅刪已驗證的清單。"""

import argparse
import csv
import io
import json
import re
from pathlib import Path

import torch
from binaryqk.history import verify_history
from binaryqk.model import ROOT, sha256

PROJECT = ROOT / "yolo_binaryqk"
OUT = PROJECT / "artifacts/migration-20261017"
KEEP_CODE = {
    "__init__.py", "model.py", "attention/__init__.py", "attention/base.py",
    "attention/quantizers.py", "variants/__init__.py", "variants/definitions.py",
    "verification/__init__.py", "verification/checkpoint.py",
}
ROOT_OLD = {
    "OPERATIONS.md", "audit_after_formal.sh", "binary_attention_executable_spec.md",
    "repair_current_t0_ema.sh", "run_formal_plan.sh", "EXPERIMENT_SEED.md",
    "IMPLEMENTATION_PLAN.md", "relative_position_bias_report.md",
}


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare():
    if (OUT / "cleanup-plan.json").exists():
        raise FileExistsError("migration already prepared")
    verified = verify_history(PROJECT)
    write("historical-reload-before.json", verified)
    evidence = {"scope": "歷史 YOLO11 結果；不代表本輪 B1～B3", "runs": {}, "reports": {}}
    # 每個 run 保留能解釋來源、失敗、精度、曲線與凍結行為的必要小型資料。
    for status in sorted((PROJECT / "artifacts/runs").rglob("status.json")):
        run = status.parent
        row = {}
        for name in ("status.json", "resolved_config.json", "validation_metrics.json",
                     "training_args.json", "parameter_delta_diagnostics.json", "attention_diagnostics.json",
                     "architecture_manifest.json", "checkpoint_manifest.json"):
            path = run / name
            if path.is_file():
                row[name] = json.loads(path.read_text())
        for name in ("training_curves.csv", "ultralytics/train/results.csv"):
            path = run / name
            if path.is_file():
                row[name] = list(csv.DictReader(io.StringIO(path.read_text())))
        evidence["runs"][str(run.relative_to(PROJECT))] = row
    for name in ("binary_attention_final_report.md", "paper_qat_selection.json", "binary_attention_summary.json"):
        path = PROJECT / "artifacts/reports" / name
        evidence["reports"][name] = json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()
    evidence["final_audit"] = json.loads((PROJECT / "logs/formal-plan/final-audit.json").read_text())
    evidence["old_method_notes"] = {name: (PROJECT / name).read_text()
                                    for name in ("binary_attention_executable_spec.md", "relative_position_bias_report.md")}
    evidence["dataset_lineage"] = {
        path.name: json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()
        for path in (PROJECT / "data").iterdir() if path.suffix in {".json", ".yaml"}
    }
    evidence["log_diagnostics"] = {}
    for path in sorted((PROJECT / "logs").rglob("*.log")):
        # 保留錯誤訊息、行數與末段；大型逐 batch 進度可由正式曲線取代。
        lines = path.read_text(errors="replace").replace("\r", "\n").splitlines()
        errors = [line for line in lines if re.search(r"Error|Traceback|failed|mismatch|repair", line, re.IGNORECASE)]
        evidence["log_diagnostics"][str(path.relative_to(PROJECT))] = {
            "line_count": len(lines), "error_lines": list(dict.fromkeys(errors)), "tail": lines[-12:]}
    write("historical-evidence.json", evidence)
    canonical = {row["sha256"]: row["path"] for row in verified}
    protected, remove = [], []
    candidates = []
    for sub in ("artifacts/runs", "artifacts/reports", "artifacts/area_metrics", "logs", "data",
                "configs", "services", "binary_attention", "tests", ".cache", ".pytest_cache"):
        path = PROJECT / sub
        if path.exists():
            candidates.extend(p for p in path.rglob("*") if p.is_file() or p.is_symlink())
    candidates.extend(PROJECT / name for name in ROOT_OLD)
    for path in sorted(set(candidates)):
        relative = str(path.relative_to(PROJECT))
        if relative == "artifacts/reports/coco_area_metrics.csv":
            protected.append({"path": relative, "reason": "yolo_optimize audit_accuracy_regressions.py reads this exact path",
                              "sha256": sha256(path)})
            continue
        if relative.startswith("binary_attention/") and relative.removeprefix("binary_attention/") in KEEP_CODE:
            protected.append({"path": relative, "reason": "minimal historical state-dict reconstruction dependency"})
            continue
        if relative == "tests/test_stage1.py":
            continue
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            # 僅刪舊 runtime link 本身，絕不跟隨資料來源。
            remove.append({"path": relative, "symlink": str(path.readlink()), "bytes": path.lstat().st_size,
                           "reason": "superseded runtime link; target retained"})
            continue
        digest = sha256(path)
        if path.suffix in {".pt", ".pth", ".safetensors"} and digest not in canonical:
            protected.append({"path": relative, "sha256": digest, "reason": "unique checkpoint; replacement not verified"})
            # 保留此權重所在 run 的重建資料。
            continue
        remove.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size,
                       "reason": "verified duplicate checkpoint" if path.suffix == ".pt" else "superseded code/runtime or extracted evidence",
                       "replacement": canonical.get(digest, "artifacts/migration-20261017/historical-evidence.json; binaryqk/")})
    # 唯一非 canonical 權重的同目錄 JSON 與 run config 仍保留原路徑。
    unique_runs = {str(Path(row["path"]).parent.parent) for row in protected if row["reason"].startswith("unique checkpoint")}
    final = []
    for row in remove:
        path = row["path"]
        if any(path.startswith(run + "/") and Path(path).suffix in {".json", ".yaml"} for run in unique_runs):
            protected.append({"path": path, "reason": "reconstruction metadata for unique checkpoint", "sha256": row.get("sha256")})
        else:
            final.append(row)
    write("cleanup-plan.json", {"root": str(PROJECT), "remove": final, "protected": protected,
                                "canonical_weights": verified, "evidence_sha256": sha256(OUT / "historical-evidence.json")})
    print(json.dumps({"remove_files": len(final), "logical_bytes": sum(r["bytes"] for r in final),
                      "unique_runs_retained": sorted(unique_runs), "protected_entries": len(protected)}, ensure_ascii=False))


def apply():
    plan = json.loads((OUT / "cleanup-plan.json").read_text())
    if plan["root"] != str(PROJECT) or sha256(OUT / "historical-evidence.json") != plan["evidence_sha256"]:
        raise ValueError("cleanup plan/evidence mismatch")
    for row in plan["canonical_weights"]:
        if sha256(PROJECT / row["path"]) != row["sha256"]:
            raise ValueError("protected canonical weight changed")
    for row in plan["protected"]:
        if row.get("sha256") and sha256(PROJECT / row["path"]) != row["sha256"]:
            raise ValueError("protected dependency changed")
    paths = []
    for row in plan["remove"]:
        path = PROJECT / row["path"]
        if not path.parent.resolve().is_relative_to(PROJECT) or ".." in Path(row["path"]).parts:
            raise ValueError("unsafe cleanup path")
        if "symlink" in row:
            if not path.is_symlink() or str(path.readlink()) != row["symlink"]:
                raise ValueError("runtime link changed")
        elif not path.is_file() or path.is_symlink() or sha256(path) != row["sha256"]:
            raise ValueError(f"cleanup candidate changed: {path}")
        paths.append(path)
    allocated = sum(path.lstat().st_blocks * 512 for path in paths if path.lstat().st_nlink == 1)
    for path in paths:
        path.unlink()
    directories = {parent for path in paths for parent in path.parents if parent.is_relative_to(PROJECT) and parent != PROJECT}
    removed_dirs = []
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
            removed_dirs.append(str(path.relative_to(PROJECT)))
    write("cleanup-result.json", {"removed_files": len(paths), "removed_empty_directories": removed_dirs,
                                  "allocated_bytes_unlinked": allocated,
                                  "note": "allocated blocks for unlinked files with link count 1; not whole-filesystem free-space delta"})
    print(f"removed {len(paths)} files, {len(removed_dirs)} empty directories; {allocated} allocated bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    apply() if args.apply else prepare()
