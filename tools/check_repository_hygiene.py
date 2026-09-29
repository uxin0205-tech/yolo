#!/usr/bin/env python3
"""檢查 Git 暫存區，避免大型訓練產物重新進入程式與報告分支。"""

from pathlib import PurePosixPath
import re
import subprocess
import sys


MAX_BYTES = 25 * 1024 * 1024
MODEL_SUFFIX = re.compile(
    r"\.(pt|pth|ckpt|safetensors|onnx|engine|torchscript|tflite|weights)(?:$|\.)",
    re.IGNORECASE,
)


def excluded_reason(path):
    item = PurePosixPath(path)
    name = item.name.lower()
    parts = item.parts
    if MODEL_SUFFIX.search(name):
        return "模型、checkpoint 或分片權重"
    dataset = (
        path.startswith("original/pose/")
        or "/artifacts/datasets/" in path
        or path.startswith("yolo_masf/bbt5-detect-baseline/dataset/")
    )
    if dataset and ("images" in parts or "labels" in parts):
        return "資料集影像／標註本體"
    if name.endswith((".log", ".pyc", ".pyo", ".cache", ".pid", ".orig", ".rej")):
        return "執行期日誌、快取或暫存檔"
    if {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}.intersection(parts):
        return "本機環境或快取"
    if name == "predictions.json" or (name.startswith("predictions_") and name.endswith(".json")):
        return "逐筆預測產物"
    if "/outputs/training/" in path and "/logs/" in path and name in {"macro.csv", "events.jsonl"}:
        return "逐步訓練追蹤"
    if re.match(r"(train|val)_batch\d+.*\.(jpg|png)$", name):
        return "batch 預覽圖片"
    if name == ".formal-plan.lock":
        return "執行期鎖檔"
    return None


def main():
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"]).split(b"\0")
    objects = {}
    failures = []
    paths = []
    for entry in entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        path = raw_path.decode("utf-8")
        paths.append(path)
        reason = excluded_reason(path)
        if reason:
            failures.append(f"{path}: {reason}")
        if stage != b"0":
            failures.append(f"{path}: 尚未解決的合併衝突")
        if mode not in {b"100644", b"100755"}:
            failures.append(f"{path}: 此分支只接受一般檔案，不接受 symlink／submodule")
        objects.setdefault(oid.decode(), []).append(path)
    request = "".join(oid + "\n" for oid in objects).encode()
    response = subprocess.check_output(["git", "cat-file", "--batch-check"], input=request)
    pointers = []
    total = 0
    for line in response.splitlines():
        oid, kind, raw_size = line.decode().split()
        size = int(raw_size)
        total += size * len(objects[oid])
        if size > MAX_BYTES:
            failures.extend(f"{path}: 單檔 {size:,} bytes，超過 25 MiB" for path in objects[oid])
        if kind == "blob" and size <= 1024:
            pointers.append(oid)
    request = "".join(oid + "\n" for oid in pointers).encode()
    payload = subprocess.check_output(["git", "cat-file", "--batch"], input=request)
    offset = 0
    for oid in pointers:
        end = payload.index(b"\n", offset)
        size = int(payload[offset:end].split()[2])
        content = payload[end + 1:end + 1 + size]
        offset = end + 2 + size
        if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
            failures.extend(f"{path}: 此分支不保存 Git LFS 指標" for path in objects[oid])
    if failures:
        print("儲存庫清理檢查失敗：", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"通過：{len(paths):,} 個檔案，邏輯容量 {total:,} bytes；無模型、資料本體或 LFS 指標。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
