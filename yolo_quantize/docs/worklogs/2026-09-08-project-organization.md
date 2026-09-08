# 專案整理：現行入口、歷史區與安全清除提案

## 使用者要求與變更

整理整個 yolo_quantize，移除首頁雜亂與過期狀態；弱證據不再放在主線結論，但保留必要負結果與血緣。依 finish-work 先盘點，再取得精確刪除授權。

主 README 重寫為現行目標、已測結果、限制、固定規則及三份必讀文件。新增 docs/CURRENT_PLAN.md，重整 reports／configs／scripts 索引，原四份入口保留到 docs/archive 並修正相對連結。新增 artifacts、程式、測試與文件閱讀說明。未移動 configs、checkpoint、run、queue 或資料集。

新增 inventory_project.py，盤點一般檔案分類及現行 JSON/YAML 引用閉包；明確禁止自動刪除。C01–C05 為可重建快取，C06 是需審慎確認的 .orig 備份；全部僅提案，等待具體授權。

## 驗證

整理前工作樹包括其他子專案修改，本次只操作 yolo_quantize 文件與 CPU 整理工具。盤點找到 834 個現行 metadata 引用路徑，解析警告 0；這不是其餘檔案可刪的證明。結果、測試、連結與機讀計畫驗證待整理結束補記。

## 困難與解法

1. 約 52 GiB 集中在 runs，歷史 V4 accepted、V35 sham、V36 parent 仍是依賴。未確認可移除的 run 全部保留，不以結果弱為由抹除證據。
2. runtime View 有大量影像 symlink。盤點改成只彙總 symlink 數，不將二十多萬筆影像路徑重複列成龐大清單；資料與連結不變。
3. 舊入口混用多個「目前」狀態。整份移至歷史區，而不是修改原實驗數字；現行只讀 CURRENT_PLAN 與本輪結果。

## 未解事項

實體刪除尚未核准；無檔案被永久清除。此整理不代表未完成的 QAT／累積配置／formal 已完成，也沒有新的 commit 或 push。既有 GPU queue 與 blocking monitor 保持運行。

## 核准後執行與驗證補記

使用者核准 C01–C05。重新核對目錄及子項均非 symlink 後，移出下列快取，保留在 `/tmp/yolo-quantize-cleanup-20260908-7ibws1zf/` 的同名 C-ID 子目錄：C01 `.pytest_cache` 34,680 bytes；C02 `.ruff_cache` 8,550 bytes；C03 `scripts/__pycache__` 67,339 bytes；C04 `tests/__pycache__` 1,266,806 bytes；C05 `src/yolo_quantize/__pycache__` 1,299,374 bytes。合計 2,676,749 bytes，減少專案快取約 2.55 MiB，但同磁碟移動不釋放實際磁碟容量。可從上述暫存恢復，/tmp 不是永久備份；快取也可重建。C06 `.orig` 保留。未刪 checkpoint、資料或 queue。

新增集中方法報告，核對 `quantizers.py` 與 `activation_adapter.py`，分清 qSiLU、LSQ+ activation 輸出及 LS-SD4 權重尺度；解釋分布假設、證據階梯與候選成本，不新增或宣稱 A-SD4 成績。舊規劃 D01/D02 已另列精確提案，待答覆，不擴張先前快取授權。

本輪 CPU 全測試 341 passed（53.13 秒）；新增整理工具及測試 lint/format 通過；入口連結檢查 20 檔、192 links、0 missing。全域 lint 有 6 個既有問題，format 有 18 檔待格式化，沒有為整理改寫可能被訓練雜湊綁定的程式。這些是明列未解項，不宣稱全域 lint 全綠。證據稽核 40 PTQ／592 probe／6 plans／3 executed graphs 通過身分一致性檢查，不能替代新 mAP 驗證。

困難：最初定位使用了不存在的 `lsq_plus.py`／`qsilu.py` 檔名；以實際模組搜尋定位修正，未因此修改實作。其他清理困難無。GPU 訓練未干預；收尾接回既有 monitor，不新增重複執行器。

最後補驗：`PYTHONDONTWRITEBYTECODE=1 ... pytest -q -p no:cacheprovider tests/test_project_inventory.py` 為 3 passed；新增方法報告與相關五份入口共 61 個本機連結，0 missing。已接回原有 blocking monitor cell 168，尚無事件返回；不是另建 GPU polling。
