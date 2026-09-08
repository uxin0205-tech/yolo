"""唯讀盤點專案及現行 metadata 引用；只產生整理／待核准清單，絕不刪檔。"""

import json
import os
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/organization"
CACHE_PATHS = [
    ".pytest_cache",
    ".ruff_cache",
    "scripts/__pycache__",
    "tests/__pycache__",
    "src/yolo_quantize/__pycache__",
]


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def reference_closure(root, seeds):
    pending = list(seeds)
    visited = set()
    references = defaultdict(set)
    warnings = []
    while pending:
        source = pending.pop()
        if source in visited or not source.is_file() or source.is_symlink():
            continue
        visited.add(source)
        if not source.is_relative_to(root) or source.suffix not in (
            ".json",
            ".yaml",
            ".yml",
        ):
            continue
        try:
            text = source.read_text()
            value = (
                json.loads(text) if source.suffix == ".json" else yaml.safe_load(text)
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            warnings.append({"path": str(source), "error": str(error)})
            continue
        for item in strings(value):
            if "/" not in item or "\n" in item or len(item) > 4096:
                continue
            path = Path(item)
            path = path if path.is_absolute() else root / path
            if not path.exists():
                continue
            references[str(path)].add(str(source))
            if path.is_file() and path.is_relative_to(root) and not path.is_symlink():
                pending.append(path)
    return {k: sorted(v) for k, v in sorted(references.items())}, warnings


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    totals = defaultdict(lambda: {"files": 0, "bytes": 0, "symlinks": 0})
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in files:
            path = Path(base) / name
            relative = path.relative_to(ROOT)
            stat = path.lstat()
            symlink = path.is_symlink()
            group = relative.parts[0]
            category = "保留／歷史或待判定"
            if group in ("src", "tests", "scripts"):
                category = "保留／程式及驗證"
            if group == "artifacts":
                category = "保留／執行狀態或實驗證據，不由盤點自動刪除"
            if group == "configs":
                category = "保留／版本化契約及血緣"
            if any(
                str(relative) == cache or str(relative).startswith(cache + "/")
                for cache in CACHE_PATHS
            ):
                category = "待核准／可重建快取"
            if not symlink:
                rows.append(
                    {
                        "path": str(relative),
                        "bytes": stat.st_size,
                        "symlink": symlink,
                        "category": category,
                    }
                )
            totals[group]["files"] += 1
            totals[group]["bytes"] += 0 if symlink else stat.st_size
            totals[group]["symlinks"] += int(symlink)
    queue = ROOT / "artifacts/queues/full-model-continuous-0907"
    seeds = [
        queue / "selected-qat-jobs-v2.json",
        queue / "parent-manifest.json",
        ROOT / "configs/experiments/full-model-continuous-0907.json",
    ]
    references, warnings = reference_closure(ROOT, seeds)
    candidates = []
    for index, name in enumerate(CACHE_PATHS, 1):
        target = ROOT / name
        if not target.exists():
            continue
        matches = [r for r in rows if r["path"].startswith(name + "/")]
        candidates.append(
            {
                "id": f"C{index:02d}",
                "path": str(target),
                "type": "可重建快取",
                "files": len(matches),
                "bytes": sum(r["bytes"] for r in matches),
                "risk": "低；測試或 import 時可能再生成，執行前須重新核對",
                "recommendation": "待使用者核准後刪除；訓練期間可先保留",
                "recoverability": "可重建，不保證原快取逐位元恢復",
            }
        )
    backup = ROOT / "tests/test_qat_runtime.py.orig"
    if backup.exists():
        candidates.append(
            {
                "id": "C06",
                "path": str(backup),
                "type": "歷史修改備份",
                "files": 1,
                "bytes": backup.stat().st_size,
                "risk": "中；不是現行 pytest 檔，但可能有唯一歷史內容",
                "recommendation": "審慎確認；未核准前保留",
                "recoverability": "未確認可靠備份，不保證可恢復",
            }
        )
    result = {
        "status": "proposal_only_no_deletion",
        "files": rows,
        "symlink_details_omitted": sum(t["symlinks"] for t in totals.values()),
        "totals": dict(totals),
        "active_reference_closure": references,
        "reference_warnings": warnings,
        "deletion_candidates": candidates,
        "limitations": "只解析現行入口可到達的 JSON/YAML 路徑，不是完整程式動態依賴證明；未列入引用不等於可刪。",
    }
    (OUT / "inventory-2026-09-08.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 整理盤點與待核准刪除清單",
        "",
        "此清單尚未執行刪除。容量為盤點快照，訓練可能持續新增檔案。",
        "",
        "## 建議清除／審慎確認",
        "",
        "| ID | 精確路徑 | 類型 | 檔數／bytes | 未使用依據及建議 | 可恢復性 | 風險 |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for c in candidates:
        lines.append(
            f"| {c['id']} | `{c['path']}` | {c['type']} | {c['files']}／{c['bytes']} | {c['recommendation']} | {c['recoverability']} | {c['risk']} |"
        )
    lines.extend(
        [
            "",
            "C01–C05 是工具生成快取，不承載模型權重或原始資料；C06 不會被 pytest 直接發現，但不代表沒有歷史價值。",
            "",
            "## 必須保留",
            "",
            "| 路徑 | 原因 |",
            "| --- | --- |",
            "| `artifacts/queues/` | 正在執行的 queue 與歷史 handoff、plan/hash 引用；不能移動 |",
            "| `artifacts/runs/` | V36 parent、V35 外部 sham、V4 accepted 指標與部署/續跑 checkpoint；未逐一證明其他 run 可刪，全部先保留 |",
            "| `artifacts/manifests/`、`configs/` | 資料/模型/方法血緣與雜湊契約；舊版本可能仍是依賴 |",
            "| `artifacts/reports/`、`deliverables/`、`docs/worklogs/` | 公開數字、負面實驗、決策與復現證據 |",
            "| `src/`、`tests/`、資料集、PDF | 實作、回歸、不可變來源與方法出處 |",
            "",
            "## 資料夾容量",
            "",
            "| 頂層 | 一般檔 bytes | 檔案／symlink 數 |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, t in sorted(totals.items()):
        lines.append(f"| `{name}` | {t['bytes']} | {t['files']}／{t['symlinks']} |")
    lines.extend(
        [
            "",
            f"現行入口引用閉包記錄 {len(references)} 個既有路徑；詳見 [JSON 清單](inventory-2026-09-08.json)。",
            "這不是未引用檔案可刪的證明；動態 import、外部專案、已發表結果與其他使用者都可能使用它們。",
            "",
            "核准時請指定 C01–C05 或個別 ID；不將「全部」擴張到未列為刪除候選的 runs／checkpoint。",
        ]
    )
    (OUT / "cleanup-proposal-2026-09-08.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "files": len(rows),
                "active_references": len(references),
                "warnings": len(warnings),
                "deletion_candidates": candidates,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
