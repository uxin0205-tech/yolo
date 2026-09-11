# Pose-head KD：已完成

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../../docs/OPERATIONS.md>)。

5 輪／1,865 macro 已完成；正式 native 對照依使用者指示取消，失敗／取消產物保留。固定全部非 Pose state，COCO／person 精確不變；最佳 E2 的 Pose AP 小升但 ball／bat 框下降，沒有合格 best_joint。

| BitTrue AP | qSiLU 起點 | head KD E2 |
| --- | ---: | ---: |
| BBAT 框 | 0.618008 | 0.613915 |
| BBAT Pose | 0.891329 | 0.892962 |
| ball 框／Pose | 0.505192／0.859649 | 0.501879／0.861716 |
| bat 框／Pose | 0.730823／0.923008 | 0.725951／0.924207 |

[完整逐輪結果](<artifacts/direct-kd-result-v1.json>)／[權重角色](<../../../reports/checkpoints/README.md>)／[全階段報告](<../../../reports/final/README.md>)。原框＋KD keypoints 的後續[推論重組](<../../inference/pose_branch_v1/README.md>)也沒有收益，不能拼接最高指標。

實際設定：AdamW、Pose head LR1e-5、warmup1、batch16、5 epochs、patience0，固定 μ10.063553373151326 對應 feature 梯度 5%。沒有本研究待執行 queue，共享 Conv 方案尚未實施。原提案與取消前配對安排保留於[歷史快照](<../../../docs/history/README.md>)。
