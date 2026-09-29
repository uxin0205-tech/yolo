---
status: accepted
date: 2026-08-28
---

# Float20 後不採用 C1～C3 並關閉下游流程

## 決策

C1、C2、C3 不納入本 revision 的後續工作。C2／C3 完整 Float 訓練、PTQ、QAT-lite 與正式
C_best 選擇永久關閉，`full35-c2-c3-auto-continuation.yaml` 固定為
`closed_rejected_by_user`，不得重新排入 GPU queue。

C0 只保留為上游 Full35 reference；這項決策不否定 `yolo_combine` 的 Full35 winner，也不影響
獨立的 `yolo_activation` 實驗。BBAT5 v1 仍是全專案唯一正式棒球資料版本。

## 理由

Float20 的 C1～C3 精度都明顯低於 C0。C2 雖然是最佳簡化候選，但 COCO mAP50-95、BBAT5
Pose box mAP50-95、keypoint mAP50-95 與 Macro F1 的下降分別為 0.220553、0.176987、
0.089216 與 0.119586，均超過原定 0.008 gate。繼續長訓練與量化無法維持本輪的精度風險邊界。

## 後果

- 保留 `results/full35-j3-float20-seed0/` 原始負結果與逐檔 hash，不回寫其中的
  `c_best=null` 或 `pending_user_decision`。
- 刪除可重建且不再使用的 checkpoint、inference副本、候選log、runtime cache與CPU驗收暫存；
  不刪除正式報告、資料 manifests 或 Portable GitHub Snapshot。
- 不宣稱本階段有 PTQ、QAT、部署 INT8 或 C_best 結果。
- 若未來要重訪架構簡化，必須建立新的專案或 spec revision、run ID 與明確授權，不能切回本封存 run。
