# 2026-10-18：Binary 第 1 階段完成、B2 選型與交付

## 任務與範圍

使用者詢問目前狀況。23:34 查詢確認第 1 階段隊列已於 **23:10:06（Asia/Taipei）**完成，GPU 使用率 0%、顯存約 611 MiB，沒有 compute process。依先前授權立即分析與收尾，使用 `finish-work` 技能由主代理核對交付；沒有啟動第 2 階段或新增 GPU 工作。

## 實際結果

三組各完成 10 epochs，完整 COCO train 118,287／val 5,000，使用共同 parent、seed 0、相同配方及 evaluator。依事前固定官方整體 AP 選型，B2 勝出。

| 候選 | 最佳 epoch | 官方 AP50–95 | internal AP50–95 |
| --- | ---: | ---: | ---: |
| B1 sign＋residual | 10 | 0.503966 | 0.497746 |
| B2 sign＋Hadamard | 10 | 0.509889 | 0.503328 |
| B3 sign＋residual＋Hadamard | 8 | 0.506355 | 0.499704 |

B2 接全模型恢復，epoch 0～5 官方 AP 依序為 0.509889、0.501803、0.492689、0.484558、0.489609、0.492613。連續 5 epochs 未超過 epoch 0，依 patience 停止。因此交付保留初篩最佳，沒有把恢復訓練稱為精度提升。原始 parent 官方 0.524686／internal 0.518002；B2 官方仍低 1.48 個百分點。歷史 optimize A0 internal 0.506739 不能與本輪官方 AP 混比。

完整 [結果報告](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/RESULTS.md)、[八項 AP／成本 CSV](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/comparison.csv)、[曲線](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/curves.svg) 及 [handoff](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/handoff.json) 已生成。隊列共約 10 小時 24 分鐘；提前停止解釋了為何早於先前跑滿 40 epochs 的 ETA。

## 變更內容與原因

1. 補齊正式報告的選型理由、原始 parent 差距、恢復負結果、八項 AP、實測訓練成本、硬體未完成項及交付血緣。
2. 同步根層 README、整合計畫、cleanup 接續狀態、Binary README、正式隊列 README 與工作紀錄索引，將進行中改為完成待檢視。
3. 修正整合計畫尾段殘留的「Q/K-only 恢復訓練」字樣，使其符合已執行的 Attention 初篩與全模型恢復；沒有更動實驗方法或程式。
4. 新增 CPU 交付驗證紀錄，確認正式權重、指標、來源與圖表可用。

## 驗證方式與結果

使用 `CUDA_VISIBLE_DEVICES=''`、本機 `.venv/bin/python`、`PYTHONPATH=yolo_binaryqk` 進行收尾核對；未重跑 GPU AP 或訓練，沒有因文件更新重跑先前已通過的 44 項 CPU 測試。

- 四份最佳 checkpoint 的 SHA、strict reload、有限數值、參數數量、scale slots 與 manifest 均通過；四份必要 resume 檔案存在，本次未重新執行 resume 訓練。
- 比對 39 次完整 val5000 評估紀錄與 35 個完整 train118287 epochs，best 選擇、CSV／JSON 及八項 AP 一致。
- B2 初篩最佳與最終 `recovery/epoch-00.pt` 的 state 逐 tensor 完全相同，immediate parent 路徑／SHA 正確；交付檔案 SHA 差異源自 phase 血緣 metadata。
- 17 份研究程式與 11 份參考來源 SHA 均與啟動時相同，包括唯讀 optimize 來源。
- 將 comparison.json 單獨放入暫存目錄，成功重畫 SVG，沒有使用 GPU 或 checkpoint。
- 收尾文件檢查：9 份文件的 139 個本機連結全部有效，沒有尾端多餘空白。

完整證據見 [handoff-verification.json](../../yolo_binaryqk/artifacts/stage1/formal-20261018-v2/handoff-verification.json)。

## 困難與解法

1. 收尾核對首次直接比較 checkpoint manifest 與 recipe JSON 時失敗：checkpoint 的 class ID 是整數鍵，JSON 物件鍵必為字串。逐項確認唯一差異為 `names` 的鍵表示，模型 strict reload、參數與數值皆正常；將比較統一為 JSON 表示後重新核對通過，沒有修改模型或訓練程式。
2. 全模型恢復未改善，且後兩輪有回升。依事前 patience 停止並保留初篩最佳；不事後延長訓練或將負結果藏掉，也不宣稱更長訓練必定無效。

## 清理與未解事項

錯誤 v1 與短測臨時權重已在先前直接刪除，清單見 [重建紀錄](2026-10-18-binaryqk-recovery-v2.md)。本次未新增刪除；現存四組 best／resume、指標、曲線與來源鎖仍承擔比較、追溯及必要續訓用途，不因結果較差就視為錯誤產物。沒有 archive／backup、commit 或發布。

下一階段建議 parent 為 B2，等待使用者檢視並指定。單 seed、僅 winner 全模型恢復、尚未恢復原始 AP、尚未完成 Q10／bias／PWL／reciprocal／qSiLU，以及無 RTL／HLS／實板量測均須保留。BBAT5 v1、Pose、唯讀 optimize 與未指定階段未修改。
