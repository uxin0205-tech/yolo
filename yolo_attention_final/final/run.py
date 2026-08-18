"""檢查、推論與重現 queue 訓練配方的單一入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from training_recipe import DEFAULT_QUEUE_ROOT, print_recipe, queue_status, run_training
from ultralytics import YOLO

from yolo_attention.attention import HardwareFriendlyAttention
from yolo_attention.config import NormalizationKind

# 區塊 1：唯一交付權重與固定模型規格。
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "pwl-final-best.pt"
EXPECTED_SHA256 = "c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123"
ATTENTION_SITES = ("model.10.m.0.attn", "model.22.m.0.1.attn")


# 區塊 2：載入權重；任何規格不符都立即停止。
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_check() -> tuple[YOLO, dict[str, object]]:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    digest = sha256_file(WEIGHTS)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 不符：{digest}")

    yolo = YOLO(str(WEIGHTS))
    model = yolo.model
    yaml = model.yaml
    sites = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, HardwareFriendlyAttention)
    }
    if yaml.get("yaml_file") != "yolo26m.yaml" or yaml.get("scale") != "m" or yaml.get("nc") != 80:
        raise RuntimeError("checkpoint 不是官方 YOLO26m、scale=m、80 classes")
    if set(sites) != set(ATTENTION_SITES):
        raise RuntimeError(f"Attention sites 不符合預期：{sorted(sites)}")

    for name, module in sites.items():
        normalizer = module.normalize
        if module.config.normalization is not NormalizationKind.BIT_TRUE_PWL:
            raise RuntimeError(f"{name} 不是 Bit-True PWL")
        if module.score.use_ste:
            raise RuntimeError(f"{name} 不應啟用 STE")
        if normalizer.score_floor != -10.0 or normalizer.segments != 20:
            raise RuntimeError(f"{name} 的 PWL 範圍或 segment 數量錯誤")
        if normalizer.endpoint_table.numel() != 21 or normalizer.endpoint_storage_bits != 336:
            raise RuntimeError(f"{name} 的 endpoint table 錯誤")

    report = {
        "weights": str(WEIGHTS),
        "sha256": digest,
        "model": "yolo26m.yaml",
        "scale": "m",
        "classes": 80,
        "attention_sites": sorted(sites),
        "normalization": "Bit-True PWL Q8.8/UQ1.15, [-10, 0], 20 segments",
    }
    return yolo, report


# 區塊 3：提供推論與完整 queue 訓練流程。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("check", help="驗證 SHA-256 與完整模型規格")
    commands.add_parser("recipe", help="顯示完整訓練階段與學習率")

    predict = commands.add_parser("predict", help="執行 Ultralytics 推論")
    predict.add_argument("source", help="圖片、目錄、影片或串流來源")
    predict.add_argument("--device", default=None, help="例如 cpu 或 0")
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--save", action="store_true")

    train = commands.add_parser("train", help="預覽 queue；--execute 才會啟動或續跑")
    train.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    train.add_argument("--execute", action="store_true")

    status = commands.add_parser("status", help="顯示重現用 queue 狀態")
    status.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    return parser


# 區塊 4：所有使用者操作都統一由這個入口分派。
def main() -> int:
    args = build_parser().parse_args()
    command = args.command or "check"
    if command == "check":
        _, report = load_and_check()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if command == "recipe":
        print_recipe()
        return 0
    if command == "predict":
        yolo, report = load_and_check()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        yolo.predict(
            source=args.source,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            save=args.save,
        )
        return 0
    if command == "train":
        return run_training(args.queue_root, execute=args.execute)
    if command == "status":
        return queue_status(args.queue_root)
    raise AssertionError(f"不支援的命令：{command}")


if __name__ == "__main__":
    sys.exit(main())
