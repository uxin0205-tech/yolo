# 2026-11-03：全部 RepConv 候選的算子簡化契約

## 任務與範圍

使用者再次要求所有 Rep 都必須經過算子簡化，包含 Softmax、activation 等。本輪把此要求固定到正式研究計畫與專案規則；尚未指定啟動 Stage8，不實作 Rep、不清理該階段資料夾，也不啟動 GPU 工作。

## 變更內容與原因

共同 parent 為已交付 Stage7 qSiLU E8，SHA `2f91ec1862ad62304f1e2727e9b9ee64a27c5a46a45137f83a74dc7f14c46a69`。三候選 Rep17、Rep20、Rep17＋20 固定使用：

- Activation：`ShiftPiecewiseQuadraticSiLU`／`qsilu_pq` 固定分段係數；新增 Rep 的 qSiLU 放在分支相加之後，folding 後仍保留。
- Attention Softmax：D1 整數 PWL exp `[-10,0]`／20 段，LUT256 reciprocal 與乘法 normalization；禁止最後再用浮點除法重正規化。
- Binary／scale／bias：B2 sign＋Hadamard、兩處四 heads、16 個 S1 Q10 scale、signed16/1024 分解式位置 bias。
- 檢查範圍：訓練前向、folding、重新載入、實際匯出圖，以及 module／functional call-sites；不能只讀 manifest 推定路徑正確。

額外的 activation／非線性、動態除法、BN、head／decode 等也須逐項盤點。可行的新簡化先做共同模型的數值與完整任務驗證，再一致套用三候選，避免與 Rep placement 比較混雜。未完成的簡化、剩餘浮點、訓練專用 surrogate、離線運算分開記錄。

唯讀參考 `yolo_optimize/model/README.md` 的 qSiLU／PWL／folded Rep 交付要求，以及現行 Stage7 handoff。此為文件需求確認，沒有採用 optimize 舊 normalization 取代本輪完整 D1；實作及訓練啟動前仍須完成對應程式、實際 run 配方與結果的完整核對。

同步更新根層 README、`AGENTS.md`、正式研究計畫與工作紀錄索引。

## 驗證方式與結果

核對既有 Stage7 handoff 與正式計畫的契約一致，三個 Rep 候選表均明列已選 D1；Stage8 狀態維持待指定。檢查本次文件差異、必要連結及空白格式。僅文件變更，不新增測試、CPU 模型運算、訓練或 AP 評估。

## 困難與解法

無。原計畫已包含硬體友善原則；本次把目前已選定的具體算子與全模型查核範圍寫清楚。

## 未解事項或風險

目前 Conv、QKV、P·V、projection 與 activation 任務路徑仍有浮點運算；硬體友善公式不代表全整數部署或已做硬體量測。其他尚未簡化算子與實際匯出支援，於 Stage8 指定開始後依實際圖查核，不預先宣稱完成。
