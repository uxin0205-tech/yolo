"""固定 YOLO26m 架構的精簡 CPU 建模與 forward smoke test。"""

from yolo_attention.cli import main

# 使用正式 Float-PWL variant 建立固定架構，確認兩個 Attention sites 都能完成 forward。
raise SystemExit(
    main(
        [
            "smoke",
            "--model",
            "yolo26m.yaml",
            "--variant",
            "configs/variants/float-pwl-final.yaml",
            "--imgsz",
            "64",
        ]
    )
)
