"""已授權的 Stage7 完成清理；先抽取證據、記錄精確範圍，再刪除產物。"""

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "yolo_activation"
OLD = PROJECT / "artifacts/runs/full35"
RESULT = PROJECT / "result/stage7"
RECORD = RESULT / "cleanup-20261103.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def inventory(path):
    counts, logical, allocated = Counter(), 0, 0
    members = hashlib.sha256()
    entries = [path]
    if path.is_dir() and not path.is_symlink():
        for base, dirs, files in os.walk(path, followlinks=False):
            entries.extend(Path(base) / name for name in dirs + files)
    for entry in entries:
        stat = entry.lstat()
        kind = "symlink" if entry.is_symlink() else "directory" if entry.is_dir() else "file"
        counts[kind] += 1
        logical += stat.st_size if kind != "directory" else 0
        allocated += stat.st_blocks * 512
        members.update((str(entry.relative_to(PROJECT)) + "\0").encode())
        if kind == "symlink":
            members.update(os.readlink(entry).encode())
    return {"path": str(path.relative_to(ROOT)), "counts": dict(counts), "logical_bytes": logical,
            "allocated_bytes": allocated, "membership_sha256": members.hexdigest()}


def main():
    if RECORD.exists():
        raise SystemExit("清理紀錄已存在，禁止重複執行；請查閱 cleanup-20261103.json")
    if read(RESULT / "handoff/validation/result.json")["status"] != "passed":
        raise ValueError("先通過可攜包驗證才能清理")
    queue = PROJECT / "artifacts/stage7/activation-20261102"
    if read(queue / "supervisor.json")["status"] != "complete":
        raise ValueError("queue 尚未完成")
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            words = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(word in (b"stage7.runner", b"scripts/run_full35_queue.py") for word in words):
            raise ValueError(f"研究程序仍執行：{proc.name}")
    deps = read(PROJECT / "stage7/external-dependencies.json")
    protected = {ROOT / name for name in deps if (ROOT / name).is_file()}
    protected.update(ROOT / name for name in read(PROJECT / "stage7/protocol.json")["sources_sha256"])
    protected.update(p for p in (PROJECT / "release/weights").rglob("*") if p.is_file() and not p.is_symlink())
    protected.update((ROOT / "configs/datasets/bbat5-v1.yaml",
                      ROOT / "original/pose/derived/bbat5-v1/configs/pose.yaml",
                      ROOT / "original/pose/derived/bbat5-v1/configs/detect.yaml"))
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    dirs, files, evidence = [], [], {}
    for base, children, names in os.walk(OLD, followlinks=False):
        base = Path(base)
        if "datasets" in children:
            directory = base / "datasets"
            if any(p.is_relative_to(directory) for p in protected):
                raise ValueError(f"資料 view 仍有受保護依賴：{directory}")
            # Runtime view manifest 很小，抽取來源；不讀連結所指的正式影像／標註。
            for manifest in directory.glob("*/manifest.json"):
                evidence[str(manifest.relative_to(OLD))] = {"sha256": sha(manifest), "data": read(manifest)}
            dirs.append(directory)
            children.remove("datasets")
        for name in names:
            path = base / name
            if path in protected:
                continue
            if path.is_symlink():
                files.append(path)
                continue
            suffix = path.suffix
            record = None
            if suffix == ".json":
                record = {"data": read(path)}
            elif suffix == ".yaml" or (suffix == ".csv" and name != "macro.csv"):
                record = {"text": path.read_text()}
            elif suffix == ".csv":
                totals = {}
                with path.open() as stream:
                    for row in csv.DictReader(stream):
                        metric = row.get("metric", "unknown")
                        item = totals.setdefault(metric, {"rows": 0, "first": row, "last": row})
                        item["rows"] += 1
                        item["last"] = row
                record = {"macro_summary": totals}
            elif suffix in {".log", ".jsonl"}:
                lines = path.read_text(errors="replace").splitlines()
                errors = [line for line in lines if any(k in line.lower() for k in ("traceback", "exception", "nonfinite", "out of memory", "error:", "nan"))]
                record = {"lines": len(lines), "first": lines[:5], "last": lines[-10:], "diagnostic_lines": errors[:100]}
            if record is not None:
                evidence[str(path.relative_to(OLD))] = {"sha256": sha(path), **record}
            files.append(path)
    write(PROJECT / "result/reference-full35/evidence.json", {"schema": "full35-compact-evidence-v1", "source": str(OLD), "entries": evidence,
          "說明": "舊結果／配置完整抽取；逐 macro／event 僅保留首尾與診斷摘要。原大型預覽圖、runtime view、快取與無用 queue 直接刪除。外部仍用檔案保留原路徑。"})
    for zero in queue.glob("zero-*/model.pt"):
        if zero.is_symlink() or zero in protected:
            raise ValueError("非預期的 zero checkpoint")
        files.append(zero)
    for base, children, _ in os.walk(PROJECT, followlinks=False):
        for name in list(children):
            if name in {"__pycache__", ".pytest_cache", ".ruff_cache"}:
                dirs.append(Path(base) / name)
                children.remove(name)
        # 上面已盤點的 runtime views 不重走八十萬個連結。
        if Path(base) == OLD:
            children[:] = []
    items = []
    for directory in dirs:
        if directory.is_symlink() or not directory.is_relative_to(PROJECT):
            raise ValueError("刪除資料夾範圍錯誤")
        items.append({**inventory(directory), "kind": "directory"})
    for path in files:
        if not path.is_relative_to(PROJECT) or path in protected:
            raise ValueError("刪除檔案範圍錯誤")
        item = {**inventory(path), "kind": "file"}
        if path.suffix == ".pt":
            item["sha256"] = sha(path)
        items.append(item)
    record = {"status": "prepared", "created_at": datetime.now().astimezone().isoformat(),
              "authorization": "使用者要求整理整個當前資料夾與交接；持續授權刪除已確認無用產物，不封存。",
              "protected_sha256": before, "compact_evidence": "yolo_activation/result/reference-full35/evidence.json",
              "items": items, "logical_bytes_removed": sum(x["logical_bytes"] for x in items),
              "allocated_bytes_removed": sum(x["allocated_bytes"] for x in items)}
    write(RECORD, record)
    print(f"已記錄 {len(dirs)} 個目錄與 {len(files)} 個檔案；開始移除", flush=True)
    for directory in dirs:
        shutil.rmtree(directory)
        print("removed " + str(directory.relative_to(PROJECT)), flush=True)
    for path in files:
        path.unlink()
    empty = 0
    for base, _, _ in os.walk(OLD, topdown=False):
        path = Path(base)
        if not any(path.iterdir()):
            path.rmdir()
            empty += 1
    after = {name: sha(ROOT / name) for name in before}
    if after != before:
        raise ValueError("受保護檔案 SHA 改變")
    record.update(status="complete", completed_at=datetime.now().astimezone().isoformat(),
                  empty_directories_removed=empty, protected_sha256_unchanged=True)
    write(RECORD, record)
    print(json.dumps({k: record[k] for k in ("status", "logical_bytes_removed", "allocated_bytes_removed", "empty_directories_removed")}), flush=True)


if __name__ == "__main__":
    main()
