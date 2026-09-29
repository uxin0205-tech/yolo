# Cleanup-v1 清理紀錄與使用邊界

## 目的與來源

依使用者要求，將 `uxin0205-tech/yolo` 整理成可承接後續更新的程式與報告基線。來源為 GitHub `main` 的 `1d409c4fe21d3287d6f19be48f4457407a6bb50e`，新分支採獨立 root commit，不繼承舊 checkpoint 歷史。既有本機工作區及未提交研究修改不納入此次同步。

保留 1,495 個既有 Python 檔、937 個 YAML，以及 Markdown、PDF、CSV／JSON 報告、統計圖表、設定和資料血緣。來源檔案共 114,933 個；原有保留項目 7,750 個，另加入維護工具與本次紀錄。演算法程式碼不改寫。

## 檔案容量與排除清單

原 main 檔案邏輯容量為 **20,636,965,087 bytes（20.637 GB）**，原有保留項目 **842,673,721 bytes（0.843 GB）**，減少 **95.92%**；另加約 20 MB 的逐檔排除紀錄及少量說明／工具。這按每個路徑計入 LFS 實體檔案，同一內容出現在多個路徑會重複計算，不能當成 GitHub 計費或實際釋放空間。

| ID | 排除內容 | 檔案數 | 邏輯大小 |
| --- | --- | ---: | ---: |
| C01 | 模型、checkpoint、分片權重 | 226 | 16.370 GB |
| C02 | 資料集影像、labels 與對應 symlink | 106,352 | 1.810 GB |
| C03 | 日誌、PID、鎖檔等執行產物 | 80 | 0.509 GB |
| C04 | 逐筆預測 JSON | 92 | 0.743 GB |
| C05 | Full35 逐步訓練 CSV／JSONL | 6 | 0.166 GB |
| C06 | train／val batch 預覽圖 | 427 | 0.197 GB |

完整路徑、分類、原 Git blob SHA 與大小見 [excluded-files.csv](excluded-files.csv)；機器可讀摘要見 [inventory.json](inventory.json)。C01 會影響續訓／推論；C02 會影響資料載入；C04～C06 會影響逐筆重算與完整舊發行包驗證。保留下來的歷史 report、manifest、checksum 不冒充完整資產包，須自行提供原資產後才能重跑。使用者原始來源與本機模型均未因建立本分支而刪除。

56 張原本採 LFS 儲存的報告圖片從本機 LFS cache 讀取，逐一核對 SHA-256 後改為一般 Git 檔案；報告圖片內容不變。所有 LFS tracking 規則已移除，CSV／SVG 證據原有的 byte-preservation 屬性保留。

## 資料集規則

此分支的 `original/` 與 architecture_2 snapshot 只保留設定、README、split、patch 與來源稽核 manifests。資料本體留在 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`，原始／歷史來源仍維持唯讀。本次沒有修改影像、labels、6,647 張樣本、5,964／683 formal split 或任何實驗結果；沒有新版本、重切分或抽樣。

## 後續使用與提交

```bash
git clone --single-branch --branch Cleanup-v1 https://github.com/uxin0205-tech/yolo.git
cd yolo
# 完成修改並選定暫存檔案後：
python3 tools/check_repository_hygiene.py
```

依各子專案 README 安裝相應環境；不同實驗的環境不一定相同。重跑前提供原權重、COCO 與 canonical BBAT5；本次沒有下載、訓練、QAT、PTQ 或重新評估精度。

根層及各子專案 `.gitignore` 已補齊 checkpoint、分片權重、快取、資料本體等規則，歷史權重例外已移除。GitHub Actions 與本機工具檢查同一暫存區政策，禁止 checkpoint、資料本體、LFS 指標及超過 25 MiB 的單檔。

## 歷史與 GitHub 空間限制

此新分支不繼承大型檔案歷史；舊 `main`、`yolo_optimize-cleanup` 及 GitHub 內部引用仍可能保留舊提交。刪除分支與 GitHub 後端回收不是同一件事。本次建立新分支不等於已刪除 main 或釋放帳號配額；分支處置結果以本次工作紀錄為準。

GitHub 官方說明：移除 LFS 指標後，遠端 LFS 物件仍計入配額。完整清除通常需要聯絡 GitHub Support，或另外刪除並重建儲存庫；後者會影響 Issues／stars／forks，不屬於本次已執行操作。參考[移除 Git LFS 檔案](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage)。

## 驗證

驗證結果與未執行項目見[中文工作紀錄](../worklogs/2026-09-29-cleanup-v1.md)。盤點與排除清單是本次發行依據，不是對歷史模型精度的新背書。
