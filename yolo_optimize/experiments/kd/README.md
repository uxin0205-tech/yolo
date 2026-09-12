# 階段 4：知識蒸餾 KD

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../docs/OPERATIONS.md>)。

| 研究 | 實際結果 |
| --- | --- |
| [dual_task_v1](<dual_task_v1/README.md>) | COCO 用 YOLO26L 教師、BBAT 用獨立 Pose 教師；K0／KD 各 5 輪，無合格新 best_joint |
| [pose_focus_v1](<pose_focus_v1/README.md>) | 固定共享／Detect，只更新完整 Pose head；5 輪完成，E2 keypoints 小升但框退化 |

兩者都由 qSiLU E2 開始，不串接退化末輪。教師不進部署模型，資料不混 class IDs。共享 Conv 小範圍適應與創新區域 KD 仍是未執行方向；沒有待執行 queue。

## AP 與部署成本補充（2026-09-12）

- [KD：九項指標與比較](<../../reports/performance/stage-7.md>)

包含 AP50–95、Params、Model size、MAC／FLOPs、Peak memory、CPU／GPU latency、target latency 及 energy/frame。target 未量測明記缺值；不將 core-only 時間當完整 pipeline。
