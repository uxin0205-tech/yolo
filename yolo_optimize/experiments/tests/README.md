# CPU 契約測試

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../docs/OPERATIONS.md>)。

本區測試共用模組的凍結／資料保護、EMA、HOG、RepConv、QK／scale 與訓練邊界。部分測試需要本機原始 source bundle 及 Python 環境，不是只 clone 報告就能執行的完整部署驗收。

目錄整理只驗證搬移工具的 root 解析、CLI help、文件連結與產物保護，不重跑無關的訓練測試或 GPU。本次只調整路徑，原測試契約不變；不把語法檢查稱為精度或硬體效能測試。
