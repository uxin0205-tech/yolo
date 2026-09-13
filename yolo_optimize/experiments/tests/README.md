# CPU 契約測試

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../docs/OPERATIONS.md>)。

本區測試共用模組的凍結／資料保護、EMA、HOG、RepConv、QK／scale 與訓練邊界。部分測試需要本機原始 source bundle 及 Python 環境，不是只 clone 報告就能執行的完整部署驗收。

cleanup 新增 `test_fixed_scale_copy.py`，只用標準庫最小物件驗證 fixed-scale patch 的 deepcopy 狀態隔離；它不取代原 `test_fixed_scale.py` 對封存 BinaryScore 的數值、校準與 state_dict 驗證。

目錄整理／文件檢查與模型回歸必須分開解讀；不把語法或 fixture 測試稱為精度或硬體效能驗證。
