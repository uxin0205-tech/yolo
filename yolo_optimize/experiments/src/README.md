# 共用研究模組

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../docs/OPERATIONS.md>)。

`yolo_optimize/` 保存早期融合後研究使用的 training、安全保護、HOG、RepConv、MASF bridge、QK、EMA 與固定 scale 模組。實際階段入口在 scripts 或各研究資料夾。

cleanup 分支另修正 `yolo_optimize/fixed_scale.py` 在 patched score 被 `deepcopy` 後，dynamic／calibration fallback 仍可能回頭呼叫原始物件的 bound-method 問題；不改 fixed coefficients、scale 數值或歷史 AP。對應隔離回歸在 [tests/test_fixed_scale_copy.py](<../tests/test_fixed_scale_copy.py>)；真實 BinaryScore 仍需原環境整合驗證。

相應 CPU 測試在 [tests](<../tests/README.md>)，報告見 [reports](<../../reports/README.md>)。
