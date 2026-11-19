# 2026-10-21：刪除 HOG 大型資源、補清 Stage2 smoke 與前階段稽核

## 任務與範圍

使用者要求完全移除這次未採用 HOG 占用的資源，並詢問前面各階段是否清過。此授權涵蓋 HOG 未採用候選及續訓檔；另依既有「無用產物直接刪除、不封存」要求，補清發現的四份 Stage2 smoke 權重。正式資料、選定主線、optimize、先前要求保留的 MASF 正式結果與必要 resume 均不刪。

使用 finish-work 的盤點、依賴核對、驗證與紀錄流程；使用者已明確授權清理，不重複請求許可。單一主代理執行，沒有使用 GPU、訓練、commit 或發布。

## 本次實際清理

先建立逐檔清單，保存路徑、類型、SHA、大小、配置區塊、理由及受保護模型狀態；確認無 HOG／MASF 活躍工作，GPU 無 compute process。每個刪除項目再核對路徑與 SHA，只刪列明檔案，不追隨符號連結。

| 範圍 | 已移除內容 | 釋放配置空間 |
|---|---|---:|
| HOG | 未採用 E1 候選、E0 副本、正式 E1 resume、候選相容連結、smoke 預覽、Python 快取、空鎖檔 | 423.52 MiB |
| Stage2 | S1／S2 各一份 smoke epoch 權重及 smoke resume，共四份 | 555.03 MiB |
| 合計 | 22 個檔案／連結，另移除 8 個空資料夾 | 978.54 MiB |

清單為 [resource-cleanup-20261021.json](../../yolo_hog/artifacts/stage5/formal-20261021-v1/resource-cleanup-20261021.json)，Stage2 另有 [清理入口](../../yolo_binaryqk/artifacts/stage2/smoke-20261019-v1/cleanup-20261021.json)。數字為被移除檔案的 allocated blocks，非整個檔案系統 free-space 差值，亦不包含前次已清的 HOG smoke 約 338.82 MiB。

HOG 清理後約 0.8 MiB，剩程式、測試、小型數據、manifest、SHA、CSV、SVG 與原主線連結；不再有 HOG 實體 checkpoint。已刪權重沒有封存，無法直接續跑原候選；若未來重新研究，須從保留的 parent 與配方重新訓練。

## 先前各階段有無清理

逐份核對歷史清單與目前檔案，不以 README 聲稱代替查核。以下為先前紀錄，不是本次重新釋放：

| 階段 | 先前已清內容 | 歷史記錄空間 |
|---|---|---:|
| 1 Binary | 舊內容、錯誤 v1 與 v2 smoke | 約 4.49 GiB |
| 2 scale／bias | 先前未清正式比較權重；本次發現四份 smoke 遺留並補清 | 本次列於上表 |
| 3 Attention | 舊 checkpoint、預覽、重複權重、快取及 KD smoke | 約 2.87 GiB |
| 4 MASF | 重建前舊內容、首輪及延長 smoke | 約 4.57 GiB |
| 5 HOG | 前一輪交付已清兩份 smoke 權重 | 約 338.82 MiB |

歷史空間依各清單原本的 file bytes／allocated blocks 口徑列出，不將其相加冒充目前可用空間。Binary 的 2,722 個舊檔、44 個錯誤 v1 檔及 8 個 smoke 檔已不存在。Attention 仍有 5 個必要相容連結；重新出現的 63 個實體項目全屬 Python／pytest／Ruff 快取，舊 checkpoint 並未恢復。MASF 舊清單重新出現的 2 個項目也都是 Python 快取；其餘舊產物不存在。各階段 smoke 清單均核對通過。

之前並不是刪光所有未選上的正式模型：Stage1／Stage2 的有效比較模型及必要 resume 仍保留；MASF 則依使用者要求，保留正式候選、結果與必要 resume，等待融合後覆核。本次未把這些有效研究模型當垃圾刪除。

## 交接與驗證

HOG handoff 的 candidate／resume 改為 null，另存歷史路徑、SHA 與不可用狀態；comparison、working-handoff 及 audit 的 retention 說明同步。歷史 AP、剝除／重載 proof 與校準數據保留，不改寫已發生的實驗結果。選定的 Stage3 KD 第五輪 checkpoint、manifest、source resume 仍存在，後續融合入口有效。

驗證全部刪除項目已不存在；主線 SHA 保持 `fe3e84cffe41c82b9e2d2ab83f6f2fdbf5df17d9cd99527ff1ef4e0d664dd71c`；20 個 MASF 正式 checkpoint 路徑與 Stage2 正式權重的 inode、大小及 mtime 均未改變。HOG 原 source.json 的所有來源 SHA 一致，交接連結無斷鏈。模型與訓練程式未變，不重跑 AP 或訓練測試。

刪除後關閉 CUDA 可見裝置，以小型 JSON 重畫圖表通過；設定不寫 bytecode，HOG 快取未重建。新增文件連結及範圍內 git diff --check 通過。最後 HOG 實體用量為 768 KiB；GPU 利用率 0%、無 compute process，約 618 MiB 既有顯示占用未處理。

## 困難與解法

前次 Stage2 收尾只保留結果，四份已完成的 smoke 權重確實未清。本次讀取 status／result，確認 smoke=true、completed，且正式 S1／S2 已完成並保留，才補刪。先前清理過的快取後來由執行 Python 重建，將它們與未刪模型區分，避免誤報前次未執行。

其餘困難：無。未變更研究方法或啟動下一階段。第六階段仍待使用者指定。
