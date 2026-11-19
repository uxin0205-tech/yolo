# GitHub階段成果發行：5090 Half 0907

## 變更與原因

依使用者明確要求上傳GitHub，commit名稱固定為 `5090 Half 0907`。發布 `yolo_quantize/` 的報告、圖表、資料摘要、CPU/PTQ證據及相關程式與設定。新增PUBLICATION_0907.yaml；原PUBLICATION_MANIFEST.yaml保留0829歷史語意。

## 驗證方式

發行前32項CPU focused regression全部通過（23.34秒），兩個report生成器ruff及format檢查通過。另檢查staged路徑、大小及來源完整性；不把此發行當作完整suite、GPU或formal驗證。

## 困難與解法

本機上層Git有大量無關變更，使用最新origin/main的隔離worktree，只複製yolo_quantize授權範圍，不改本機訓練工作樹。完整盤點生成器需要未公開checkpoint及runs作hash稽核，故發布既有輸出快照並清楚註記；weight-evidence附錄可由公開CPU profile與CSV重建。

diff whitespace檢查報出CSV的CRLF、Markdown硬換行、Matplotlib SVG路徑空白及歷史文件末尾空行；為保留實驗證據hash，新增子專案.gitattributes定義上述格式的whitespace規則，不重寫原始證據數值或位元内容。

## 未解事項與風險

上傳後複核發現一個歷史備份 `tests/test_qat_runtime.py.orig` 被帶入，追加同名發行提交從Git快照排除並新增ignore；不改寫已推送歷史。本機該備份原件保留，僅移除此次建立的隔離發行副本。

新四天queue、formal及全整數部署仍未完成，未因上傳而啟動GPU。快取清理仍未獲授權，因此不刪本機檔案；僅排除發行。訓練checkpoint與data不隨此提交公開。此發行保留本機程式研究狀態，不宣稱全部歷史路徑已驗證。
