# 第三階段交接：D1＋KD 第五輪 EMA

正式交接是 [kd-epoch05](kd-epoch05/README.md) 的 `evaluated.pt`，官方 COCO AP50–95 **0.514620791**。使用者覆核接受 KD E5，已從 resume 匯出 EMA、CPU 嚴格重載逐位相等，再完整評估 COCO val5000；八項 AP 與 E5 紀錄完全一致。後續 Stage4～9 已完成，總入口見 [result/](../../../result/README.md)。

[handoff.json](handoff.json) 是本階段的當前來源；`user-selection.json` 記錄使用者取代原自動 +0.001 門檻的理由。`kd-selection.json` 保存原自動決策，不能當成現行選擇。

## 零加訓與 KD 結果

原 D1 使用整數 PWL exp `[-10,0]`／20＋LUT256 reciprocal，由 S1 直接替換後 AP **0.514189349**。原 `evaluated.pt` 與完整 manifest 保留為 KD 必要 parent。KD 完成 5E，E5 比 D1 提高 0.000431442，未達原 +0.001 增益門檻，但八項 AP 的退化保護通過；之後由使用者覆核採用。

- [normalization／reciprocal 完整比較](../../artifacts/stage3/formal-20261019-v1/RESULTS.md)。
- [KD 報告](../../artifacts/stage3/kd-20261019-v1/RESULTS.md) 與 [恢復／KD 判斷](../../docs/stage3-recovery-kd.md)。
- `metrics.json`／`profile.json`／`result.json` 是零加訓 D1 的原始結果；KD E5 指標應讀 `handoff.json` 及 `kd-epoch05/`，不能混讀。

本階段已停止。2026-11-17 整理時，KD 的 `artifacts/stage3/kd-20261019-v1/training/resume.pt` 因 HOG source guard 仍核對其 SHA 而保留。部署或下游接續使用已匯出的 E5 checkpoint。沒有新增訓練、硬體綜合或權重發布。
