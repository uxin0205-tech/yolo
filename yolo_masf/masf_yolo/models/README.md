# 模型模組

`builder.py` 建立 YOLO11m P2 slots；`mfam.py` 實作 MFAM/PartialMFAM；`selective.py` 實作 SP2 Ball-only P2 head；`transfer.py` 做語意權重轉移。P3M 是 P3 的 DW3×3 + DW5×5 + DW(1×7→7×1)，沒有 9×9。SP2 只簡化 prediction tower，P2 feature map 仍存在。

本套件沒有 Hadamard／雙 basis attention；`basis_lambda` 與 learnable thresholds 位於同層另一個 `yolo_binaryqk` 專案。
