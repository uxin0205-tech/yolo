# 2026-10-03：記錄工作起點與 COCO 球棒／球評估目標

## 任務與範圍

使用者詢問開始工作時會從哪裡著手，並要求未來評估加入 COCO 的球棒與球；目前先不執行，只納入目標。本次限於更新本機正式研究計畫、README 與工作紀錄，沒有啟動研究階段或評估。

## 變更內容與原因

- 正式計畫沿用 `docs/research/yolo-research-plan-2026-09-21.md`，修訂日期更新為 2026-10-03，保留 2026-09-22 初次保存日期。
- 說明工作起點：先核對本機狀態、Detect parent／checkpoint SHA、既有結果及資料／evaluator 口徑，唯讀參考 `yolo_optimize`，再準備第 1 階段 `yolo_binaryqk` 的 B1～B3。使用者另指定階段時依其範圍執行。
- 增加跨階段 COCO Detect 目標：保留整體與 person 評估，另外分別報告 `baseball bat` 與 `sports ball` 的 bbox AP50、AP50–95、Precision、Recall，以及影像／標註數和相對 parent 的變化。
- 固定沿用既有完整 COCO2017 validation split，執行前核對類別映射與 evaluator。此目標補充既有八項主要 AP，不自行改寫選型門檻或增加訓練任務。
- COCO 類別結果與 BBAT5 v1 的 ball／bat Detect、Pose 結果分開呈現；未改 canonical `/home/uxin/yolo/original/pose/derived/bbat5-v1/`、來源、影像、labels 或 split。
- 同步更新 Combine 與最終成果的交付描述、根 README 及工作紀錄索引；所有研究階段仍為「待指定」，新增評估目標為「待實作／待評估」。

## 驗證方式與結果

- 唯讀核對本機 `coco2017.yaml`：YOLO 類別索引 34 為 `baseball bat`、32 為 `sports ball`。COCO annotation category ID 的對應留待實作前核對，不把連續索引直接當成 category ID。
- 檢查四份本次修改／新增文件的連結、行尾空白與 code fence；確認階段狀態未升格，B1～B3 與既有研究順序不變。
- 本次是計畫與文件更新，未新增程式測試，未載入模型、讀取權重內容、推論、訓練、執行 COCO／BBAT5 評估或啟動 GPU 工作；未提交或推送 GitHub。

## 困難與解法

一般沙箱讀取命令遇到 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`；透過工具權限審查後執行必要唯讀檢查，再使用 patch 更新指定文件。其他困難：無。

## 未解事項或風險

本次尚未確認實際使用的 parent 權重、各模型是否已有可用分類別原始結果，以及後續正式評估的預算與實作位置；這些在指定階段開始時核對。沒有產生新的精度結果。

COCO `sports ball` 按 COCO 原類別定義報告，不能把結果直接稱作 BBAT5 棒球精度。所有模型比較仍須維持同一資料、類別映射與評估口徑。

入口：[正式研究計畫](../research/yolo-research-plan-2026-09-21.md)、[根 README](../../README.md)。
