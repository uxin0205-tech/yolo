# Checkpoint 與模型導航

索引不複製權重，全部指向 experiments 下的新位置，權重內容不變。不要直接把自訂 state-dict 交給原生 YOLO 載入；需使用對應 source 重建 qSiLU、MASF 及 PWL，並維持 weights_only 安全載入。

| 角色 | 原始模型 | 判斷 |
| --- | --- | --- |
| default_joint | 權重（本機／歷史參照：`../../experiments/activation/bridge_v1/artifacts/runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt`；未隨本次報告發布） | 目前新研究預設；activation-relative，非原嚴格 gate 全過 |
| prefusion_masf | 權重（本機／歷史參照：`../../experiments/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/masf-e8-bittrue.pt`；未隨本次報告發布） | 使用者選定融合前 P3 bridge E8 |
| prefusion_control | 權重（本機／歷史參照：`../../experiments/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/control-e8-bittrue.pt`；未隨本次報告發布） | 融合前無 MASF 配對 |
| pose_recovery | 權重（本機／歷史參照：`../../experiments/combine/bridge_v1/artifacts/fusion/j3-pose-head-recovery-v1/inference/best_pose.pt`；未隨本次報告發布） | activation 父模型 |
| pose_kd_candidate | 權重（本機／歷史參照：`../../experiments/kd/pose_focus_v1/artifacts/runs/kd-e5-seed1-v1/inference/best_pose.pt`；未隨本次報告發布） | headKD E2；框退化，不升版 |
| mixed_keypoint_candidate | 權重（本機／歷史參照：`../../experiments/inference/pose_branch_v1/artifacts/branch-v2/candidate.pt`；未隨本次報告發布） | 推論重組負結果，不升版 |

[六個模型的來源／SHA／續訓檔索引](<registry.json>)；[全部 406 個封存模型產物 CSV](<../final/checkpoints.csv>)。406 包含外部來源、不同 bank 與研究候選，不是 406 個全部合格模型。

有 training snapshot 不等於任意中斷 exact resume 已驗證。原始舊 combine 仍在 `/home/uxin/yolo/yolo_combine/final/full35/`，不與本輪 qSiLU 主線混名；完整比較見[總報告](<../final/README.md>)。
