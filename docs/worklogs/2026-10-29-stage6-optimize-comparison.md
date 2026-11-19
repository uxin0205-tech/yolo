# 2026-10-29：KD 恢復進度與 optimize 最終模型比較

> 後續：第六階段已完成本輪交付，optimize 也已使用相同 batch 與 evaluator 重驗；最新結果見 [2026-10-30 完成紀錄](2026-10-30-stage6-results.md)。下方保留當時快照。

查核時間：2026-10-29 21:45（Asia/Taipei）。完成 KD 第 2 輪，正在第 3／5 輪、240／463 macro。GPU PID 2600265 正常運行，600 秒背景監測正常，兩個完成 epoch 均未觸發安全線。第 1 輪一次 AMP overflow 已由既有機制重試成功，第 2 輪沒有重試；沒有 OOM／NaN 中止。

## 比較對象與口徑

此次以 optimize 根 README 指向的 `reports/current-model/CURRENT.json` 為準：使用者於 2026-09-15 選定的 **僅 Pose MASF、Rep17＋20 E5**。不是 9 月 11 日 qSiLU E2，也不是更早的舊 combine。實際權重 SHA 與原 run 的 independent_validation 已核對一致。

本輪固定採最新完整 KD E2 同一 checkpoint 的全部指標，不拼接不同 epoch 的最高值。下表均為 Ultralytics 內部 AP50–95，COCO 取本輪同次驗證保存的 internal 欄位。先前回報的本輪官方 COCO AP 為 0.512676；不能拿它直接和 optimize 的內部 AP 比。

| 指標 | optimize 最終 Rep17＋20 E5 | 本輪 KD E2 | 差異（百分點） |
|---|---:|---:|---:|
| COCO 整體 | 0.505869 | 0.505687 | -0.0183 |
| COCO person | 0.626986 | 0.624764 | -0.2222 |
| BBAT 框 | 0.609797 | 0.605234 | -0.4563 |
| BBAT 關鍵點 | 0.891041 | 0.894992 | +0.3951 |
| 球框 | 0.492397 | 0.486577 | -0.5819 |
| 球關鍵點 | 0.858876 | 0.855794 | -0.3082 |
| 球棒框 | 0.727198 | 0.723891 | -0.3307 |
| 球棒關鍵點 | 0.923205 | 0.934190 | +1.0985 |

目前並非全部較差：COCO 整體接近，BBAT 整體與球棒關鍵點較高；person、框偵測與球關鍵點仍偏低。尚未全面追平，也尚未選定最終融合模型。與更早舊 combine（BBAT 框 0.630036、Pose 0.903717）相比，這兩個整體值仍較低，但那不是 optimize 目前最後選定的版本。

本次 KD 已將 COCO baseball bat 官方 AP 從 J1 E6 的 0.462670 恢復至 0.470841；BBAT box／pose 由 0.605645／0.895802 到 0.605234／0.894992，整體接近原起點，不能把前兩輪稱為全面恢復。

## 驗證、修改與後續

唯讀核對實際程序、GPU、supervisor heartbeat、完整 epoch metrics、optimize 現行選定索引、權重 SHA、原 run summary／設定與 evaluator。只新增比較快照、中文紀錄並更新 README／計畫入口；沒有改執行中的程式、配方、來源鎖、資料或模型，也沒有另啟 GPU 工作。

每輪含驗證約 23 分鐘；依目前進度預估今晚 22:40～22:50 完成五輪與選定權重重驗。背景 queue 保持運行，第七階段未啟動。

## 困難與限制

舊總報告曾將 qSiLU E2 列為預設，但後來已有 9 月 15 日人工選定；已追查最新 CURRENT.json，避免拿歷史起點冒充最終成果。另有官方 COCO／內部 AP 的算法差異，已改用同類指標。

兩邊皆完整 canonical val、imgsz640、FP32 E2E，但本輪 validation batch8，歷史 Detect32／Pose16；矩形 padding 可能受 batch 分組影響，尚未用完全相同 evaluator／batch 重新跑兩份模型，千分位差異不視為顯著。當前仍是 Stage6 原生 SiLU、無 MASF／Rep，optimize 已含 qSiLU、Pose MASF、folded Rep；這是可用模型比較，不能歸因單一方法。

[數據、SHA 與程序快照](../../yolo_combine/artifacts/stage6/recovery-kd-20261029/progress-review-20261029-2144.json)
