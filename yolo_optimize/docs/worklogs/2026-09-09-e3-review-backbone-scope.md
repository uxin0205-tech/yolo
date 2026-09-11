# 2026-09-09：E3 結果、E5 延長與 Backbone 授權

## 變更與原因

使用者允許必要時修改 Backbone 並重新訓練。本分支仍只處理融合前 COCO80，overall／person 為唯一決策 AP；融合後研究完整保留。允許解凍不等於已換架構，先完成目前成對恢復的短期趨勢判斷。

## E3 實測結果

兩臂正常完成，均每 epoch 使用 118287 張訓練影像、925 次 optimizer 更新，並驗證完整 5000 張影像。以下是 EMA BitTrue internal AP50–95（0–1）；不是 COCO 官方 JSON evaluator AP。

| 權重 | overall | person |
| --- | ---: | ---: |
| 官方 FP 基準 | 0.518019276 | 0.630794912 |
| A0 原始 BinaryQK | 0.506738574 | 0.626805274 |
| control E3 | 0.506635073 | 0.626101753 |
| QK 修復 E3 | 0.505981519 | 0.626092575 |

QK E3 比 control overall 低 0.000653553，person 低 0.000009178；相對 A0 為 -0.000757055／-0.000712699。QK overall 由 E2 的 0.506345468 回落，person 則由 0.625838623 回升；沒有一致增準趨勢。梯度接通已驗證，但尚未證明能補回 BinaryQK 缺口，更不能將整個精度缺口歸因於梯度斷點。

## 下一階段

先保持相同訓練範圍與超參數，從各自完整 E3 延續至 E5，避免第三個 epoch 的波動立即觸發結構變更。AdamW、Q/K LR 5e-7、head LR 2.5e-5、warmup 僅原 E1、physical32×4、10-epoch cosine／criterion horizon 不變。保留 optimizer、scaler、EMA age；續訓採新的成對 deterministic 資料流，不聲稱無中斷 replay。

若 E5 仍無收益，下一個候選是解凍 attention 所在 block／Backbone 後段，讓 V、輸出投影及上游特徵適應 BinaryQK。需先列出精確 layer／參數範圍，驗證梯度與更新比例，低 LR 校準後再做單變因對照。必要時才擴大到全 Backbone 或替換其結構；不把換結構、改 optimizer、加 MASF 同時進行，也不預設全 Backbone 一定更好。HOG／RepConv／MASF 優化保留，但不在根基尚未恢復時疊加。

## 驗證方式

擴充 `continue_a0.py` 只接受已稽核完整 E1／E3 邊界：檢查歷史 epoch 連續、每回合影像／macro 數、optimizer 與 EMA step、criterion 保存時點；保留既有 source_run，patience 從既有歷史重建。新增 `run_pair_e5.py`，先兩臂各做一個真實 macro 的 E3 邊界 smoke，通過 optimizer／EMA 2775→2776、criterion updates=3、相同新資料 trace，才串行啟動正式 E4–E5。smoke 不納入正式訓練來源。

CPU 檢查已確認原始 head 的 reg_max=1、dfl=Identity，沒有 DFL 固定卷積參數被 head 解凍規則誤開。此檢查不代表所有 BN 或 AP 問題已排除。

## 困難與解法

原 v2 僅允許 E1，不能直接把 E3 檔名交給同一程式假裝續訓；本次擴充邊界稽核，禁止 smoke 或部分 epoch 快照。原 summary 的 source_run 重寫與 patience 從末 epoch 重算也一併修正，保持歷史可追溯。沒有新增資料、沒有改 sibling 原始碼或原始模型。

## 未解事項與風險

E5 邊界驗證及正式訓練結果待 queue 回報，不能把排程當成果。尚無通過驗收的候選；單 seed 短訓可能不足以判別微小 AP 差異。Backbone 新範圍與 LR 必須經校準後才能訓練。實際影片尚未提供；後續仍需選定 checkpoint 的可載入推論與同圖比較。

## 續訓啟動補記

兩支變更程式的 AST 語法檢查通過；2026-09-09 13:17（Asia/Taipei）兩臂真實 E3 邊界 smoke 均正常退出，queue 的 optimizer／EMA 2775→2776、criterion updates=3、相同新資料 trace、歷史 source_run 斷言全部通過。證據保存於 `studies/pre-fusion-full35-b100/artifacts/continuation-proof-v3.json`。smoke 權重不作正式來源。

13:17:38 正式啟動 control E4–E5，接續 QK E4–E5；monitor session 76299，每次 shell wait 最多 600 秒。正式兩臂結果尚待完成事件；正常等待期間不讀進度 log 或額外查 GPU。此次續訓驗證無新增困難。
