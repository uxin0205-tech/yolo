# 階段 5：推論處理

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../docs/OPERATIONS.md>)。

| 實驗 | 結論 |
| --- | --- |
| [pose_branch_v1](<pose_branch_v1/README.md>) | 原框分類＋KD E2 keypoints：框精確保留，但 Pose 無增益，不升版 |
| [routing_v1](<routing_v1/README.md>) | one2many＋NMS：bat 明顯提升、ball 下降，不全面採用 |

這兩項都是既有權重的推論驗證，不是新訓練。預設仍是原 qSiLU one2one；ball／bat 分流 routing 尚未實作、未量測雙分支成本。每個子目錄包含程式、參數、原始 metrics 與 README，未合併覆寫。

## AP 與部署成本補充（2026-09-12）

- [推論：九項指標與比較](<../../reports/performance/stage-8.md>)

包含 AP50–95、Params、Model size、MAC／FLOPs、Peak memory、CPU／GPU latency、target latency 及 energy/frame。target 未量測明記缺值；不將 core-only 時間當完整 pipeline。
