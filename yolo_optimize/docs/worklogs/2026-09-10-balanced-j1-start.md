# 2026-09-10：依實驗 4 校準任務梯度並啟動 J1

## 校準結果

`task-weight-calibration-v1` 正常完成，16 個校準 macro 的已加權 Pose／Detect 共享 norm ratio 中位數為 5.8558084（Pose weight0.25）。依 target ratio1 推導並以0.005為間隔取整，候選固定 weight0.045；後續8個 training macro 確認中位數1.2762651，落在事先設定0.5–2範圍。不是 validation 搜參，也不是每張影像動態調整。

每 macro 恢復同一 checkpoint，optimizer LR=0，未保存更新模型；總6144 Detect／384 Pose training images，來源split未變。當前合成梯度量級失衡已可透過較小 Pose weight 緩和，但未證明 AP 已提升。觀察 cosine 多接近零、部分負值，不據此直接加入 PCGrad。

## J1 新設定

`balanced_joint.py` 從 merge J0 最佳來源（SHA256 `42c48372c451e87112e63399acc064608088322711d1f311af9c2cb51c3a2c19`）開始；原 `merge-j1-v1` 未執行，避免重複正式訓練。

AdamW、Detect weight1／Pose weight0.045；20 epoch、patience8、warmup1。Neck LR7.5e-6、Detect／Pose head LR2e-5、MASF含α LR1e-6；backbone與attention固定、shared及MASF BN統計固定，heads BN按原combine政策訓練。640、Detect logical128／physical32，每macro256 Detect＋16 Pose；不改資料或 loss 公式。

## 分離最終驗收與訓練安全

原始來源與已完成結果不改，只有新 run 採以下事先記錄政策：

- 最終 checkpoint 選擇：COCO overall／person 相對原独立 Detect 最多下降0.005；其他六項 AP 相對原独立 Pose 最多下降0.02（依PDF最終精度要求，比舊final0.08更嚴格）。未通過不升格，即使訓練跑完也不算成功。
- 訓練中災難性下降：COCO任一項低於原Detect超過0.05，或六項BBAT相對本次初始化下降超過0.08，先保存再停止。
- 前4個epoch容許有限暫時適應；從第5個epoch起，COCO若仍低於原Detect超過0.02，保存停止。這是新工程策略，不是PDF原封不動的規則，也不是保證5epoch必然恢復。

`training_safety` 已做明確邊界回歸：E1下降0.03不立即停止、E5仍下降0.03停止、任何epoch超過0.05的COCO下降／0.08的Pose下降停止，E5下降0.004不停止。原最終COCO gate仍0.005，未被訓練寬限取代。

## 驗證與啟動

`balanced-j1-smoke-v1` 真實兩個joint macro完成：MASFα與context梯度非零／有限、α確實更新、固定live／EMA參數與MASF BN、硬體契約通過。正式 `balanced-j1-v1` 已啟動；事件檔 `combine/artifacts/logs/balanced-j1-v1.events.jsonl`，600秒blocking monitor。尚未有正式J1精度結果，activation／方向2未開始。

## 困難與未解

啟動修復：`balanced-j1-v1` 在寫 resolved config、任何 optimizer 更新之前，因直接以不相容 Session 實例呼叫帶 `super()` 的方法而 TypeError。已改正繼承鏈，保存 callback 明確呼叫正式保存，再套用新的訓練安全政策，避免意外重啟舊每輪0.005立即停止。CPU 回歸通過實際 `_resolved_config` JSON 序列化與 mocked 真實保存回呼：E1 person下降0.03可經新寬限規則，不觸發舊gate。保持已通過的GPU更新smoke，不重複無關測試。失敗run保留，正式重啟為 `balanced-j1-v2`；事件檔相應為 `balanced-j1-v2.events.jsonl`。

困難是舊報告權重0.25不適用目前梯度量級，且此前把最終gate用作首epoch停止可能過早截斷適應；依使用者PDF證據調整，未宣稱找到所有根因。固定weight0.045在後期是否仍合適要看訓練中既有梯度樣本與完整AP，不主動頻繁查log。沒有覆寫／刪除原模型、沒有commit或push。
