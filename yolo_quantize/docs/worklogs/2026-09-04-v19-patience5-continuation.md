# 2026-09-04：V19 Epoch 9 邊界切換 patience=5

## 變更內容與原因

- 依使用者指示，等待當時的 Epoch 9 完成訓練、Bit-True validation 與 last.pt 寫入後，才停止原本 patience=0 的 V19 QAT 程序。
- 原始 configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml 保持不變，避免既有 plan SHA-256 與 Epoch 0–9 artifacts 失去可重現性。
- 將 Epoch 9 邊界 last.pt 建立不可變硬連結：
  - checkpoints/epoch-0009-before-patience5.pt
  - SHA-256：ca700f72a699ea338b13fd6708a9c48ea6cce5e8f13ed53055e02d35da691996
- 新增 hash-pinned sidecar：
  - configs/experiments/v19-poly-shift-all-w8-qat-patience5-continuation-v1.yaml
  - 固定 base plan、resume checkpoint、Epoch 0–9 joint-score 歷史、邊界與使用者授權。
- 新增 qat_continuation.py，僅允許在完整 epoch checkpoint 上遷移 patience：
  - 驗證 base plan 與 checkpoint SHA-256。
  - 驗證 checkpoint 位於 formal_epoch_end 且下一 epoch 為 10。
  - 將既有 joint-score 歷史重播為 best_score=0.8317651484021804、stale_epochs=4。
  - 保留模型、EMA、optimizer、scheduler、AMP scaler、criterion、RNG 與 selector state；只更新 resolved patience contract 與 early-stop state。
- qat_runtime.py 與 qat_queue.py 新增 sidecar CLI、有效 patience、合法 early-stop completion 驗證及既有 best checkpoint 彙整。
- 後續尚未執行的 QAT 規劃統一改為 patience=5：
  - V25 nine-region quality QAT。
  - V28 Paper-TWN short QAT recipe。
  - Full35 整合 roadmap 的後續短／延伸 QAT 預設。

## 實際停止邊界

- Epoch 9 checkpoint 已完整保存，progress.next_epoch=10、joint_epochs_completed=10、global_macro_step=4630。
- 原程序在寫完 Epoch 9 checkpoint 後剛執行 Epoch 10 的 macro 0；該未完成 macro 不屬於 epoch-boundary snapshot，停止後由 Epoch 9 邊界精確續跑。
- 舊 queue 正確留下 queue_failed、return code -2，作為人工授權切換的歷史事件，不偽裝成正常完成。

## patience 語意

- Epoch 5 是目前最佳 joint score。
- Epoch 6、7、8、9 均未改善，因此續跑前 stale_epochs=4。
- Epoch 10 若仍未改善，將達到 patience 5；runtime 會先完成 validation 與 checkpoint，再正常 early stop。
- 若 Epoch 10 改善，stale 計數歸零，可繼續到最多 15 epochs。
- Epoch 5 的 gate-green best_joint.pt 不會因後續退化而被覆蓋。

## 驗證方式與結果

- python -m py_compile：qat_continuation.py、qat_runtime.py、qat_queue.py 通過。
- 相關 CPU 測試：19 passed in 17.15s。
- Ruff lint：全部通過。
- 真實 sidecar CPU preflight：
  - ready=True
  - effective_patience=5
  - resolved_j3_patience=5
  - best_score=0.8317651484021804
  - stale_epochs=4
  - blockers：無
- 未改動 COCO 或 BBAT5 資料、split、影像及標註。
- GPU 續跑啟動確認：
  - queue PID 947935、QAT 主程序 PID 947994。
  - migrated checkpoint 為 resume-patience5-epoch0010.pt，SHA-256 為 09f7816e91c01fe5ac3312173a9bd3addc43a34d9caa2222996ef215b7a065f4。
  - runtime sidecar 顯示 patience=5、stale_epochs=4。
  - Epoch 10 已前進至 macro 7/463；AMP scale 8、overflow retries 0，Detect/Pose/shared gradients 皆存在。

## 遇到的困難及解法

- apply_patch 因 sandbox loopback 錯誤無法讀取工作區；後續以同內容的 unified diff 與系統 patch 套用。
- 一次大型 hunk 被系統 patch 截斷；在 GPU 尚未重啟前立即以自動產生、行數正確的完整函式 diff 修復，並以 py_compile、pytest、Ruff 三重確認。
- 已刪除過程產生的 .orig 與 .rej 暫存檔；未刪除任何實驗 checkpoint 或使用者資料。

## 完成結果

- queue 已正常完成，總計完成 11 個 epoch（Epoch 0–10）；Epoch 10 驗證後因連續 5 個 epoch 未改善而依 `patience=5` 正常 early stop。
- 最佳 checkpoint 仍為 Epoch 5 的 `best_joint.pt`，SHA-256：`61f70406e5d5412530298ad26d585dd856942e68830be127891fd61e0877936e`。
- Epoch 5 通過全部 gate；相對 parent 的 COCO box mAP50、COCO person mAP50、COCO box mAP50-95、COCO person mAP50-95 分別為 -0.014133、-0.009638、-0.012950、-0.009933。
- Epoch 10 未通過 COCO detection gate：上述四項 delta 分別為 -0.056893、-0.025082、-0.047797、-0.028050；BBAT box mAP50 為 +0.002953、box mAP50-95 為 -0.019676、pose mAP50 為 +0.003920、pose mAP50-95 為 +0.003697。
- 結論：延長訓練未超越 Epoch 5，且解凍後主要退化集中於 COCO detection；後續應鎖定 Epoch 5 作 V19 QAT 候選，再與 nine-region W8 PTQ parent 做完整比較，不以 Epoch 10 作交付模型。
- queue 與 QAT 程序皆已退出，GPU 已釋放。

## 未解事項或風險

- V19 已完成；尚待針對 Epoch 5 與 nine-region W8 PTQ parent 做鎖定分析，才決定下一個 GPU job。
- Epoch 10 顯示 full-model unfreeze 後 COCO detection 漂移，下一輪 recovery 需評估較長 scale-only、較低 model LR 或更窄的 trainable scope，且必須維持 matched sham。
- V19 的原 matched sham 跑滿 15 epochs；patience 只縮短 QAT 的最大實際執行長度，不改 Epoch 0–9 的 optimizer 或資料順序。completion manifest 會明記 planned epochs、effective patience、early-stop event 與 continuation lineage。
