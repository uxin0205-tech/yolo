# CPU 回歸測試

測試覆蓋格式、圖轉換、QAT、資料／血緣、queue、指標判定與報告工具；不等同 mAP 或 GPU 吞吐量實驗。

日常命令見[腳本說明](../scripts/README.md)。執行完整測試時使用 `CUDA_VISIBLE_DEVICES=-1` 並限制 CPU threads，不佔用 GPU。

`__pycache__` 是可重建快取，列於[清除提案](../docs/organization/cleanup-proposal-2026-09-08.md)；`test_qat_runtime.py.orig` 是歷史備份，不被 pytest 直接執行，尚未核准刪除。
