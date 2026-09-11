# 共用研究模組

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../docs/OPERATIONS.md>)。

`yolo_optimize/` 保存早期融合後研究使用的 training、安全保護、HOG、RepConv、MASF bridge、QK、EMA 與固定 scale 模組。實際階段入口在 scripts 或各研究資料夾。

本次只補導航，僅隨 experiments 搬移，不改模組訓練邏輯。相應 CPU 測試在 [tests](<../tests/README.md>)，報告見 [reports](<../../reports/README.md>)。
