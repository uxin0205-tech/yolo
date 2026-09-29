# 唯讀清理建議

本次依使用者要求沒有刪除、搬移或重新整理任何 runtime artifact。以下只記錄日後可人工審核的候選，不代表已授權執行。

| 候選 | 目前用途 | 建議時機 | 可恢復性與風險 |
|---|---|---|---|
| `final/**/__pycache__/`、`.runtime/matplotlib/` cache | Python／Matplotlib 可再生 cache | 任何需要縮小工作目錄時 | 可完全重建，低風險；不影響 manifest（已排除 pycache） |
| `artifacts/runs/*/ultralytics/weights/last.pt` | 同一 run 的斷電恢復 | 所有報告、best、Bit-True 與遠端備份完成後 | 可由本機 best 保留結果，但不能恢復 exact optimizer state；刪前須再確認不續跑 |
| 多個 `source-accepted-a2-*-bittrue.pt` | 不同 queue 的 A2 來源封裝證據 | Git／外部備份已驗證後 | state-dict 相同但封裝 SHA 不同；若刪除會失去逐 queue provenance，建議至少保留 `checkpoint-inventory.json` 與一份原檔 |
| 舊 `patience7-r1` 失敗 run | 舊 1.5 GiB RAM floor 的中斷證據 | 問題鑑識與報告不再需要時 | 不可由完成 run 重建原始失敗現場；目前建議保留 |
| `artifacts/validation/*/ultralytics/predictions.json` | canonical COCO AP 可重算來源 | canonical JSON、checkpoint 與 evaluator 契約均有雙重備份後 | 可重新 inference 產生，但成本高；目前建議保留 |
| `original/pose/dataset/`、`detect_dataset/` 與 zip | 使用者資料與衍生驗證集 | 不屬於本工作項目清理範圍 | 可能是唯一資料來源，禁止自動刪除 |

實際清理前應先驗證 `final/MANIFEST.json`、Git LFS 遠端物件與 checkpoint SHA256，並再次取得使用者對明確路徑的刪除授權。
