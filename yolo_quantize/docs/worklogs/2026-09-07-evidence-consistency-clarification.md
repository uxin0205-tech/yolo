# 歷史與本輪比較條件釐清

## 變更及原因

依使用者指出的 parent 不一致與 CPU 數量誤讀風險，新增證據分層報告與可重建 CPU 血緣稽核。舊兩份盤點／分布報告與交付入口加歷史 snapshot 提示，原數字不改寫；本輪結論以 V36 同源產物及雙門檻報告為準。

## 驗證

本輪 40 組 PTQ 實際 build、592 組輸出探測、6 份 QAT 計畫與 3 份已執行圖的 parent/export/full-resume EMA 血緣核對通過。148 層各 4 格式覆蓋通過；明確確認 probe 的 map_validation=false。新增雜湊／activation 不一致拒絕測試，連同精度彙整測試共 9 passed（0.02 秒），ruff 通過。沒有 GPU 操作，原 queue 與 blocking monitor 保持運行。

## 困難及解法

舊 reference、mAP50-only selection 及 warm-start 的 v19 校準標籤容易誤導。保留既有產物與 hash，不直接覆寫；在新報告明列實際來源欄位、V36 雜湊與現行雙門檻入口，避免資料血緣與顯示名稱混為一談。

## 未解事項

未重跑歷史 1,332 筆為 V36；尚未完成 148 層逐層 mAP 矩陣，也不宣稱全部格式逐層 QAT。有限候選的 mAP/QAT 與累積配置驗證持續進行，純整數部署與 formal 未完成。此稽核不證明歷史排序可遷移，亦不把 QAT 全模型適應當成單層隔離因果證據。
