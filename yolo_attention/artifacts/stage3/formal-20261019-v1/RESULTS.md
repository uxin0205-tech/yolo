# 第 3 階段評估與選型

全部候選從同一 S1 Q10 parent 接續；沒有 exact Softmax 正式對照組或 control 加訓。
N1～N9 先做共同 parent 的兩處 score/P/PV 診斷，再依事前門檻評估。D1～D3 固定 BitTrue PWL exp。

| 候選 | 官方 AP50–95 | 相對 parent | 評估狀態 |
| --- | ---: | ---: | --- |
| N1 | 0.514166 | +0.000014 | 完整 COCO val5000；無加訓 |
| N2 | 0.514205 | +0.000053 | 完整 COCO val5000；無加訓 |
| N3 | 0.514159 | +0.000007 | 完整 COCO val5000；無加訓 |
| N4 | 0.514213 | +0.000061 | 完整 COCO val5000；無加訓 |
| N5 | 0.513654 | -0.000498 | 完整 COCO val5000；無加訓 |
| N6 | 0.510462 | -0.003689 | 完整 COCO val5000；無加訓 |
| N7 | 未執行 | — | 超過事前 PV relative L2 診斷門檻，未做完整 AP |
| N8 | 未執行 | — | 超過事前 PV relative L2 診斷門檻，未做完整 AP |
| N9 | 未執行 | — | 超過事前 PV relative L2 診斷門檻，未做完整 AP |
| D1 | 0.514189 | +0.000038 | 完整 COCO val5000；無加訓 |
| D2 | 0.514129 | -0.000023 | 完整 COCO val5000；無加訓 |
| D3 | 0.514169 | +0.000017 | 完整 COCO val5000；無加訓 |

完整路徑候選推薦：D1；選型狀態：complete。
採用門檻為官方 AP 與 parent／最佳比較候選的差距各不超過 0.001；完整路徑內選 AP 最高，完全同分才比 ROM。
若門檻未通過，recommendation 只是待處理候選，不產生正式 handoff；先診斷或完成其他 normalization 的整數分母，再考慮少量適應訓練。

D1：256-entry reciprocal LUT；D2：16 段 reciprocal PWL；D3：64-entry LUT＋一次 Newton。
reciprocal 使用 Q1.16 mantissa 與獨立 exponent，probability 為 UQ1.24。沒有最後浮點重正規化。
exp 為 Q8.8→UQ1.15，[-10,0]/20，exp(0)=1；最大 1024 tokens 的分母需要 uint26。
三組均保留 S1 的 binary Q/K、Q10 scales、signed16 bias、activation call-sites；載入不允許默默回退。
八項官方 AP、internal AP、檔案大小與解析 ROM 見 comparison.csv；每個 candidate 的 metrics／profile／checkpoint 可追溯。
QKV、P·V、projection、Conv、SiLU 仍為浮點；這是定點 reference，沒有 RTL／HLS 或板端實測加速。
只用一個既有 seed parent，沒有 control 加訓；16 張真實資料診斷不能取代完整 val5000 或代表所有輸入。
第 4 階段未啟動。

## 完成交付與恢復判斷

選定 D1 已實體整理到 [result/stage3](../../../result/stage3/README.md)，舊 D1 目錄為相容連結。九個正式權重嚴格重載通過，資料與 parent 一致；CPU 193 passed／1 skipped、Ruff 通過，圖表以單一 comparison.json 在無 GPU／checkpoint 的臨時目錄重畫成功。

本階段無需恢復訓練；[KD 條件方案](../../../docs/stage3-recovery-kd.md) 已評估但未啟動。原始教師 overall／person／ball 較高，bat 反而較低，不能保證 KD 全面改善。GPU 已釋放，第 4 階段未啟動。
