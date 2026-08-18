"""可直接執行的 queue 訓練與選擇方法。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# 區塊 1：定位 production 專案並指定新的 immutable queue。
FINAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FINAL_ROOT.parent
DEFAULT_QUEUE_ROOT = PROJECT_ROOT / "artifacts/final-reproduction-queue"
PROJECT_CLI = (sys.executable, "-m", "yolo_attention.cli")


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
    source = str(PROJECT_ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else source + os.pathsep + current
    return environment


def _call(*arguments: str) -> int:
    command = (*PROJECT_CLI, *arguments)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=_environment(), check=False).returncode


def _required_inputs() -> tuple[Path, ...]:
    return (
        PROJECT_ROOT / "src/yolo_attention/cli.py",
        PROJECT_ROOT / "configs/variants/float-pwl-final.yaml",
        PROJECT_ROOT / "configs/variants/bittrue-pwl-final.yaml",
        PROJECT_ROOT / "configs/evaluation/coco2017.yaml",
        PROJECT_ROOT / "data/coco2017.yaml",
        PROJECT_ROOT / "weights/v1-br-best.pt",
        PROJECT_ROOT / "artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt",
    )


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
