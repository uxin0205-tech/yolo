"""可直接執行的 queue 訓練與選擇方法。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# 區塊 1：final 本身就是 production 專案，不依賴上一層目錄。
FINAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FINAL_ROOT
DEFAULT_QUEUE_ROOT = PROJECT_ROOT / "artifacts/final-reproduction-queue"
PROJECT_CLI = (sys.executable, "-m", "yolo_attention.cli")
EXPECTED_CHECKPOINT_SHA256 = {
    "pwl-final-best.pt": "c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123",
    "weights/v1-br-best.pt": "9e2ca0d93793785f0f7e514d876580204070bac57b4163d7c5f0c5ca352b9c3f",
    "artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt": (
        "2f5a18e37eca6f780e1aa52c9aeb50306e4aba7f14e1c22a8b499bcf8a40d8ac"
    ),
}


# 區塊 2：以單一資料結構完整呈現實驗配方。
TRAINING_RECIPE = {
    "fixed_model": "官方 yolo26m.yaml、scale=m、80 classes",
    "training_normalization": "可微分 Float-PWL",
    "formal_evaluation": "完整 5,000 張 COCO2017 val 的 Bit-True PWL",
    "common": {
        "seed": 0,
        "optimizer": "AdamW",
        "batch": 16,
        "imgsz": 640,
        "workers": 8,
        "amp": True,
        "deterministic": True,
        "scheduler": "constant",
        "warmup_epochs": 0,
        "warmup_bias_lr": 0,
        "weight_decay": 0.0005,
        "selection_metric": "mAP50-95",
    },
    "stages": [
        {
            "name": "block LR x1/x2/x4 掃描",
            "max_epochs": 8,
            "patience": 3,
            "attention_lr": [5e-6, 1e-5, 2e-5],
            "adjacent_block_lr": [1e-6, 2e-6, 4e-6],
        },
        {
            "name": "Neck/Detect 恢復",
            "max_epochs": 16,
            "patience": 5,
            "lrs": {"attention": 5e-6, "adjacent_block": 1e-6, "neck_detect": 5e-7},
        },
        {
            "name": "Backbone 最後 stage 恢復",
            "max_epochs": 16,
            "patience": 5,
            "lrs": {
                "attention": 5e-6,
                "adjacent_block": 1e-6,
                "neck_detect": 5e-7,
                "backbone": 1e-7,
            },
        },
        {
            "name": "全模型恢復",
            "max_epochs": 20,
            "patience": 6,
            "lrs": {
                "attention": 5e-6,
                "adjacent_block": 1e-6,
                "neck_detect": 5e-7,
                "backbone": 1e-7,
            },
        },
    ],
    "gate": "評估實際 Bit-True best.pt；child 比直接 parent 低超過 0.001 就回退",
    "global_selection": "選最高 Bit-True 候選；訓練權重要成為正式 winner，三-seed mean 必須提升 0.001",
    "batchnorm": "recovery 全程鎖定所有 running mean、variance 與 counters",
}


# 區塊 3：呼叫 repository 中已測試的訓練引擎，不複製實作。
def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(PROJECT_ROOT)
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else source + os.pathsep + current
    return environment


def _call(*arguments: str) -> int:
    command = (*PROJECT_CLI, *arguments)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=_environment(), check=False).returncode


def _required_inputs() -> tuple[Path, ...]:
    return (
        PROJECT_ROOT / "yolo_attention/cli.py",
        PROJECT_ROOT / "configs/variants/float-pwl-final.yaml",
        PROJECT_ROOT / "configs/variants/bittrue-pwl-final.yaml",
        PROJECT_ROOT / "configs/evaluation/coco2017.yaml",
        PROJECT_ROOT / "data/coco2017.yaml",
        PROJECT_ROOT / "weights/v1-br-best.pt",
        PROJECT_ROOT / "artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt",
    )


def _load_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML 必須是 mapping：{path}")
    return payload


def _write_mapping(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def configure_runtime(
    *,
    data_root: Path,
    device: str,
    batch: int,
    workers: int,
) -> dict[str, object]:
    """寫入本機資料與運算設定；已初始化 queue 時拒絕改寫。"""

    if batch < 1 or workers < 0:
        raise ValueError("batch 必須大於 0，workers 不得小於 0")
    if (DEFAULT_QUEUE_ROOT / "queue.json").is_file():
        raise RuntimeError("重現 queue 已初始化；為保留 provenance，不允許再改寫訓練設定")

    data_root = data_root.expanduser().resolve()
    required_directories = (
        data_root / "images/train2017",
        data_root / "images/val2017",
    )
    missing = [str(path) for path in required_directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError("COCO2017 目錄不完整：" + ", ".join(missing))

    dataset_path = PROJECT_ROOT / "data/coco2017.yaml"
    dataset = _load_mapping(dataset_path)
    dataset["path"] = str(data_root)
    _write_mapping(dataset_path, dataset)

    training_paths = sorted((PROJECT_ROOT / "configs/training").glob("*.yaml"))
    for path in training_paths:
        recipe = _load_mapping(path)
        recipe.update(
            {
                "data": "data/coco2017.yaml",
                "device": str(device),
                "batch": batch,
                "workers": workers,
            }
        )
        _write_mapping(path, recipe)

    evaluation_path = PROJECT_ROOT / "configs/evaluation/coco2017.yaml"
    evaluation = _load_mapping(evaluation_path)
    evaluation.update(
        {
            "data": "data/coco2017.yaml",
            "device": str(device),
            "batch": batch,
            "workers": workers,
        }
    )
    _write_mapping(evaluation_path, evaluation)
    return {
        "data_root": str(data_root),
        "device": str(device),
        "batch": batch,
        "workers": workers,
        "training_recipes_updated": len(training_paths),
        "evaluation_recipe": str(evaluation_path.relative_to(PROJECT_ROOT)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def doctor_report() -> dict[str, object]:
    """檢查交付檔案、checkpoint hashes 與資料集目錄，不啟動 GPU 工作。"""

    missing_inputs = [
        str(path.relative_to(PROJECT_ROOT)) for path in _required_inputs() if not path.is_file()
    ]
    checkpoint_hashes: dict[str, dict[str, object]] = {}
    for relative, expected in EXPECTED_CHECKPOINT_SHA256.items():
        path = PROJECT_ROOT / relative
        actual = _sha256_file(path) if path.is_file() else None
        checkpoint_hashes[relative] = {
            "expected": expected,
            "actual": actual,
            "ok": actual == expected,
        }
    dataset_path = PROJECT_ROOT / "data/coco2017.yaml"
    dataset_root: Path | None = None
    dataset_missing: list[str] = []
    if dataset_path.is_file():
        raw_root = Path(str(_load_mapping(dataset_path).get("path", ""))).expanduser()
        dataset_root = raw_root if raw_root.is_absolute() else PROJECT_ROOT / raw_root
        dataset_root = dataset_root.resolve()
        for relative in ("images/train2017", "images/val2017"):
            if not (dataset_root / relative).is_dir():
                dataset_missing.append(relative)
    return {
        "portable_root": str(PROJECT_ROOT),
        "required_inputs_ok": not missing_inputs and all(item["ok"] for item in checkpoint_hashes.values()),
        "missing_inputs": missing_inputs,
        "checkpoint_hashes": checkpoint_hashes,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "dataset_ok": dataset_root is not None and not dataset_missing,
        "dataset_missing": dataset_missing,
        "queue_root": str(DEFAULT_QUEUE_ROOT),
        "queue_initialized": (DEFAULT_QUEUE_ROOT / "queue.json").is_file(),
    }


# 區塊 4：預設 dry-run；只有 --execute 才會初始化或續跑 GPU 工作。
def print_recipe() -> None:
    print(json.dumps(TRAINING_RECIPE, indent=2, ensure_ascii=False))


def run_training(queue_root: Path, *, execute: bool) -> int:
    queue_root = queue_root.expanduser().resolve()
    missing = [str(path) for path in _required_inputs() if not path.is_file()]
    preview = {
        "execute": execute,
        "queue_root": str(queue_root),
        "recipe": TRAINING_RECIPE,
        "missing_inputs": missing,
    }
    if not execute:
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0 if not missing else 1
    if missing:
        raise FileNotFoundError("缺少訓練輸入：" + ", ".join(missing))

    if not (queue_root / "queue.json").is_file():
        code = _call(
            "queue",
            "init-pwl-lr-sweep",
            "--project-root",
            str(PROJECT_ROOT),
            "--queue-root",
            str(queue_root),
        )
        if code:
            return code
    code = _call("queue", "validate", "--queue-root", str(queue_root))
    if code:
        return code
    return _call("queue", "run", "--queue-root", str(queue_root), "--execute")


def queue_status(queue_root: Path) -> int:
    queue_root = queue_root.expanduser().resolve()
    if not (queue_root / "queue.json").is_file():
        print(json.dumps({"queue_root": str(queue_root), "status": "尚未初始化"}, ensure_ascii=False))
        return 1
    return _call("queue", "status", "--queue-root", str(queue_root))
