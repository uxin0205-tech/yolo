# 2026-09-10：完整 Pose 首輪保護停止與 LR 單變因試驗

## 事件與結果

`full-pose-adamw-v1` 首 epoch／373 macro 後觸發 POSE_SAFETY_STOP，先保存 checkpoint 再正常退出，非完成 60 epoch。BBAT box AP 0.517259543、Pose AP 0.833610914；ball box／pose 0.411923391／0.781413956；bat box／pose 0.622595696／0.885807872。

相對起點，ball box -0.063927427、ball pose -0.047911576，超過預設 0.03 保護；bat pose 雖上升，但不足以接受。圖內診斷 COCO AP 0.220813708，不代表外部原 Detect checkpoint 被改動；顯示新 Pose trunk 尚不能直接接回原 Detect head。

## 受控下一步

假說：完整解凍起始更新過大；可證偽預測是同起點、同資料、seed、scope、BN、loss horizon 下，全部 role LR×0.1 可減輕首 epoch 退化。這不是已確認根因，亦可能與 loss schedule 重啟或特徵／統計適應相關。保持 AdamW，不同时更換 optimizer 或放寬 gate。

新增 `full_pose_gentle.py`，backbone 1.5e-6、Neck 7.5e-6、Pose head 2e-5；其餘沿用原試驗（60 epoch、warmup 1、patience 12、任一 BBAT AP 比起點下降超過 0.03 保存停止）。原 run 保留，新結果在 `artifacts/fusion/full-pose-gentle-v1/`。

真實兩 batch smoke 通過，backbone／Neck／head 更新、固定 Detect／EMA 與硬體契約均通過，結果在 `full-pose-gentle-smoke-v1/summary.json`。正式受控 run 已啟動，事件檔 `combine/artifacts/logs/full-pose-gentle-v1.events.jsonl`；600 秒 blocking monitor。

## 困難與未解

困難是完整 Pose 解凍初期各類表現分歧，不能以 bat keypoint 上升掩蓋 ball 下降。較低 LR 是否有效尚未驗證，不能稱為已修復。完成／停止後需比較首 epoch 同口徑與完整結果，再決定下一步；不直接融合。沒有覆寫、刪除、commit 或 push。
