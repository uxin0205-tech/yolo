# 六組短 QAT 結束後：有限的累積 PTQ

本計畫的十二候選已完成。依 2026-09-08 使用者與老師的新決定，量化延後；下文「最多兩組 QAT」只保留為原規劃，不再是目前啟動授權。結果與未完成事項見[階段收尾](2026-09-08-quantization-phase-handoff.md)。

六組各 5 epochs 已完成；[完整結果](2026-09-07-continuous-qat-recovery-results.md)保留所有回合與 16 指標。Detect predictor LS-SD4 在 PTQ 不過門檻時，第 2–5 回合恢復到門檻內；第 5 回合 COCO／Person mAP50–95 相對 PTQ 回升約 4.192／4.461 百分點。這證明該配置可恢復，不保證所有 PTQ 失敗皆可恢復。

## 起點與成本控制

繼續使用已通過 export 重驗的 V36 parent，與先前 592 probe／40 PTQ 保持同起點。六組 QAT 用於提名格式，不拼接它們的獨立參數，也不聲稱目前 PTQ 已繼承六組訓練的精度。後續需要時才對累積模型短 QAT。沒有重新訓練 baseline。

新增 **最多 12 個 PTQ 候選**，不新增訓練；後續恢復 QAT **總共最多再兩組，各 5 epochs**。四天仍為軟目標。新的機讀契約明確收緊此階段上限，不沿用舊契約的無總上限欄位來無限追加。

## 順序與提名

| 步驟 | 區域／配置 | 本次新增替換 |
| --- | --- | --- |
| 1 | backbone early | `graph.model.3.conv` → SD4 |
| 2 | backbone deep | `graph.model.5.conv` → SD4 |
| 3 | backbone attention safe | `graph.model.10.cv1.conv` → SD4 |
| 4 | neck | `graph.model.20.conv` → SD4 |
| 5 | neck attention safe | 5 個安全路徑 → SD4 |
| 6 | MASF | 3 個路徑 → filterwise TWN |
| 7 | Detect tower | `graph.model.23.detect_head.one2one_cv3.2.0.1.conv` → SD4 |
| 8 | Pose tower | 24 個路徑 → SD4 |
| 9 | Detect predictor | 6 個路徑 → SD4，保留 QAT 恢復可能 |
| 10 | Pose predictor | `graph.model.23.pose_head.one2one_cv3.2.2` → Paper-TWN |
| 11、12 | W6／W5 兩個平行替代配置 | 對同一已接受前綴中尚未改動、原 bit 較高的 backbone early/deep／neck 路徑降低 bit |

第 1–4、7、10 步利用既有 raw boxes／scores／kpts 的單層 probe：在誤差最低前 25% 的候選中，選 nominal 位元節省最大者；排除原本已用相同或更低 bit 的無收益投影。這是限制搜尋成本的提名規則，非最優性證明，不使用 Top-300 後處理輸出或特徵圖的大範圍數值替代 raw 排名。細項數字見[機讀計畫](../../artifacts/queues/full-model-cumulative-0908/plan.json)。

第 5、8、9 步使用同區既有 PTQ／QAT 依據；第 6 步的 filterwise TWN 在末回合整體指標较好，但 Person 非最佳，仍須累積重驗。PTQ 中 SD4 尺度固定；不把這一步稱為已學習的 LS-SD4，必要的尺度學習在後續 QAT。

## 保留規則與可追溯性

每個候選都從 V36 export 重新構造「已接受路由＋本步提名」，避免重複投影上一層、避免跨模型拼權重。實際評估的是累積模型，單層隔離證據仍是之前的 probe；不能把此批稱為全 148 層隔離 mAP。

只有 16 個總精度門檻全過、且包含尺度的權重碼儲存估算確實小於前一步，才鎖定新路由；否則保留前一步已量化配置，負結果留下。成本不含 bias、protected、alignment／kernel 開銷，不報虛構部署加速。W6/W5 兩者從相同前綴分叉，不能先把 W6 勝出結果當 W5 起點。

原始報告仍有歷史 mAP50-only selection，但此 queue 不使用它晉級；重新要求全部 8+8 項指標。NaN／缺指標／雜湊不一致為錯誤事件，不以退回上一層掩蓋。一般精度未達標則是可用負結果，不是執行錯誤。

未接受提名的其餘層依然保留 parent 量化，148 部署權重路徑覆蓋不變，qSiLU＋LSQ+ A8 不變，protected 邊界不變。這不是全模型純整數部署完成報告。

## 執行入口與驗證

新 queue：[使用說明](../../artifacts/queues/full-model-cumulative-0908/README.md)。先 CPU 準備、再執行 11 個階段（最後階段兩候選）；完成後狀態 `decision_required`，由結果挑最多兩個累積 QAT，不宣稱已經排上尚未選定的訓練。

CPU：351 tests passed；新 runner 與測試 lint／format 通過；第一階段真實 schema smoke 通過；六組 QAT 實際 graph、40 PTQ、592 probe 同源稽核通過。後續階段的具體前綴依前步結果決定，會在該階段進 GPU 前固定計畫並做 schema 檢查。
