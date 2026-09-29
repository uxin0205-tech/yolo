# 2026-09-11：雙教師完成後的 Pose-only 分析與下一步規劃

## 變更與原因

使用者詢問可否暫時不訓 person、只訓 Pose，要求分析與下一步決策。新增 `kd/dual_task_v1/POSE_ONLY_NEXT_PLAN.md`，決定優先凍結全部非 Pose state、只更新完整 Pose head；不新增 GPU job，不把分析請求當新訓練啟動。

## 驗證方式與結果

主代理依 diagnosing-bugs 技能重播既有 K0／KD 五輪 Bit-True JSON，E5相對起點0.001 gate 明確 FAIL；比較完整曲線，KD E4六項BBAT平均0.753921仍低於起點0.754668。E4 Pose overall +0.002792，但 ball box -0.009751，因此不能只忽略 person 就判成功。

讀取現有 SpatialRouter 接線，確認 KD tap 在 layer16／19／22；當共享層固定，此 loss 對 Pose head 梯度為0。新計畫改為 PH0原生／PH-KD head內訊號的同起點配對，head KD 實作與真實更新驗證尚待執行。先使用 qSiLU E2，不以退化 E5 或 E4 shared權重作起點。

## 困難與解法

無執行錯誤。限制是現有實驗未分離兩教師的共享梯度影響，也未證明 head-only 能克服共享表示瓶頸；以三個可檢查假說及最少兩臂計畫處理，不宣稱已找出唯一原因。

## 未解事項與執行狀態

K0／KD正式各5輪已完成，無新研究 queue。新計畫為 AdamW、head LR1e-5、warmup1、batch16、各5輪；若有收益才評估延長。需先完成 head KD tap／梯度／BN／EMA／safe export 前置。此回合只分析與規劃，未訓練、未改 checkpoint、未刪除或發布結果。
