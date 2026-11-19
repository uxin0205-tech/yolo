# 2026-10-19：第 2 階段完成與交付核對

## 任務與範圍

使用者詢問目前狀況時，確認 S1／S2 已於 2026-10-19 04:58（Asia/Taipei）完成。兩組各五輪，共約 4 小時 23 分鐘，沒有超出事前預算。依先前授權完成結果分析、CPU 交付驗證及文件同步；未啟動第 3 階段，未使用 GPU 再評估或加訓。

## 結果與建議

| 模型 | 官方整體 AP50–95 | Person AP50–95 | Ball AP50–95 | Bat AP50–95 |
| --- | ---: | ---: | ---: | ---: |
| B2 parent | 0.509889 | 0.639523 | 0.514560 | 0.487651 |
| S1 Q10 | **0.514152** | 0.643376 | 0.512632 | 0.485961 |
| S2 二次冪 | 0.513711 | **0.644395** | 0.512593 | 0.473084 |

兩組最佳皆在第 5 epoch。按事前「官方整體 AP 最高，完全同分才優先 shift」規則推薦 S1。S1 較 B2 提高 0.4263 個百分點，較 S2 高 0.0441 個百分點；單 seed 差距很小，不能宣稱統計顯著或硬體成本也較佳。S2 使用 shift，保留為硬體成本比較，但不依此改寫事前主線選型規則。

整體及 person 改善，ball／bat 沒有改善；S1 的 ball／bat 各比 B2 低 0.1928／0.1689 個百分點，S2 的 bat 低 1.4567 個百分點。此結果仍是 COCO80，不是 BBAT5 專用模型結論。S1 官方整體 AP 距原始 native parent 0.524686 尚低約 1.0534 個百分點。

每組參數 21,897,272，比 B2 多 1024 個（16 個 scale、1008 個 bias entries）；checkpoint 約 88.08 MB。每張 640×640 的 2.56M 次尺度計算中，S1 用固定整數乘法，S2 用固定 shift。兩組 bias 表各 2016 bytes、score 累加 signed18。Q/K 二值化保留，native Softmax、SiLU、QKV／P·V／projection 仍未全定點；沒有硬體綜合或板端加速量測。

完整八項 AP、小型數據、圖表、部署常數、parent 與 SHA 見 [正式報告](../../yolo_binaryqk/artifacts/stage2/formal-20261019-v1/RESULTS.md)、[comparison.csv](../../yolo_binaryqk/artifacts/stage2/formal-20261019-v1/comparison.csv) 與 [handoff.json](../../yolo_binaryqk/artifacts/stage2/formal-20261019-v1/handoff.json)。

## 驗證方式與結果

核對來源 SHA、兩組共同 recipe／parent、六次完整 val5000、五輪完整 train118287、每輪 463 optimizer steps、最佳指標與 checkpoint SHA、十項必要參數的實際更新、resume SHA、部署常數及每 600 秒監測。實際最佳權重以嚴格 loader 在 CPU 重建，再跑 64×96 與 96×64 的有限輸出檢查；這只是載入及 shape smoke，不是額外 AP 或第 9 階段形狀實驗。

驗證結果保存於 [handoff-verification.json](../../yolo_binaryqk/artifacts/stage2/formal-20261019-v1/handoff-verification.json)。原先 Binary 50 項、Attention 168 項通過／1 項略過，以及兩組 GPU 短測仍見 [verification.json](../../yolo_binaryqk/artifacts/stage2/verification.json)；本次沒有修改訓練程式，不重跑無關測試。

上述交付核對全部通過。兩組監測各 14 次，最長間隔約 600.13 秒，沒有異常動作；實際權重重建與兩種 shape 前向皆輸出有限值。另在只有 comparison JSON 的臨時資料夾成功重畫 SVG，沒有 GPU 或 checkpoint；八份完成文件的本機連結檢查通過。

## 困難、清理與未解事項

困難：無。所有正式 epochs 的 AMP retries 為 0，監測未記錄失敗或 stale child 終止。查詢時 GPU 無 compute process、利用率 0%，只剩約 622 MiB 系統／桌面占用。

清理：本次沒有刪除。S1／S2 都是有效正式結果；最佳 checkpoint、必要 resume、比較及負結果保留，不因 S2 未入選就將其視為錯誤。切換第 3 階段時依使用者要求，將 S1 選定資料夾整理到 `result/stage2/`，再由 `yolo_attention` 接續；目前仍停在第 2 階段等待檢視。

未解事項／風險：單 seed、沒有未改方法 control 加訓，不能把收益完全歸因於量化 scale 或 bias；S2 成本可能更低但未硬體量測。ball／bat 未改善需在後續任務分析保留，不得只報整體提升。第 3 階段 PWL／reciprocal、後續 qSiLU／RepConv 均未啟動。
