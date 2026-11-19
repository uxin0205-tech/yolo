"""固定 YOLO26m 架構的精簡 CPU 建模與 forward smoke test。"""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from yolo_attention.cli import main

# 使用正式 Float-PWL variant 建立固定架構，確認兩個 Attention sites 都能完成 forward。
raise SystemExit(
    main(
        [
            "smoke",
            "--model",
            "yolo26m.yaml",
            "--variant",
            str(WORKSPACE_ROOT / "configs/variants/float-pwl-final.yaml"),
            "--imgsz",
            "64",
        ]
    )
)
