# 雙教師 KD：已完成

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../../docs/OPERATIONS.md>)。

qSiLU E2 為起點，K0／空間 KD 各 5 輪已完成。KD E4 有 bat Pose 收益，但 ball 框下降，沒有新的合格 best_joint；原起點不被替換。

- [方法、teacher、μ、超參數與 gate](<PLAN.md>)
- [後续 Pose-only 的推導與原計畫](<POSE_ONLY_NEXT_PLAN.md>)
- [實際結果與全階段比較](<../../../reports/final/README.md>)
- 原始資料：artifacts/runs/、artifacts/teacher-validation-v1/、artifacts/training-queue-v1/。

MuSGD 在本起點校準未過，主線使用 AdamW；不外推為 MuSGD 普遍無效。`run_training_pair.py` 是已完成實驗入口，不是現在應直接啟動的 queue。舊說明見[歷史快照](<../../../docs/history/README.md>)。
