# 模型模組

`builder.py` 建立 YOLO11m P2 slots、B1R per-scale Detect 與 B0 三尺度 P3 parent；`mfam.py` 實作 Phase 1 legacy MFAM/PartialMFAM，以及第二輪採用論文式 (1)–(6) 的 `PaperFormulaMFAM`/`PartialPaperFormulaMFAM`；`selective.py` 實作 SP2 Ball-only P2 head；`transfer.py` 做語意權重轉移。P3 第二輪是原始 P3 C3k2 後接 MFAM，不替換 C3k2 本身，也不含 9×9 的非指定變體。SP2 只簡化 prediction tower，P2 feature map 仍存在。

`configs/models/yolo11m-p2-slots-b1r.yaml` 是 B1R 專用 template：P5 output capped at 512，讓 B0 的 P3/P4/P5 Detect tensors 可完整承接；原本 Phase 1 的 `yolo11m-p2-slots.yaml` 保持不變。

本套件沒有 Hadamard／雙 basis attention；`basis_lambda` 與 learnable thresholds 位於同層另一個 `yolo_binaryqk` 專案。
