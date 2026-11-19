# combine 式獨立 Pose 完成比較

官方 yolo26m-pose 初始化；固定 bbat5-v1 formal val 683 張；batch8，按解析度分開比較。
歷史960成績引用既有同設定評估；本次新增三階段640／960及舊Stage2／3／4的640評估。

| 模型 | 解析度 | Box mAP50–95 | Pose mAP50–95 | 平均 |
| --- | --- | --- | --- | --- |
| combine_recipe_p1_640 | 640 | 0.23220 | 0.56895 | 0.40057 |
| combine_recipe_p2_640 | 640 | 0.58872 | 0.89492 | 0.74182 |
| combine_recipe_p3_640 | 640 | 0.63445 | 0.91701 | 0.77573 |
| stage2_adamw_640 | 640 | 0.62348 | 0.91343 | 0.76846 |
| stage3_adamw_960 | 640 | 0.58995 | 0.88870 | 0.73933 |
| stage4_musgd_960 | 640 | 0.59786 | 0.88879 | 0.74333 |
| combine_recipe_p1_640 | 960 | 0.19896 | 0.50752 | 0.35324 |
| combine_recipe_p2_640 | 960 | 0.58578 | 0.87478 | 0.73028 |
| combine_recipe_p3_640 | 960 | 0.61408 | 0.90788 | 0.76098 |
| stage1_adamw_640 | 960 | 0.58332 | 0.90225 | 0.74279 |
| stage2_adamw_640 | 960 | 0.59519 | 0.89700 | 0.74609 |
| stage3_adamw_960 | 960 | 0.61322 | 0.90847 | 0.76085 |
| stage4_musgd_960 | 960 | 0.62433 | 0.90507 | 0.76470 |

640最佳：combine_recipe_p3_640，權重 /home/uxin/yolo/yolo_pose_training/runs/combine_recipe_p3_640/weights/best.pt。
960最佳：stage4_musgd_960，權重 /home/uxin/yolo/yolo_pose_training/runs/stage4_musgd_960/weights/best.pt。

每階段best.pt、last.pt、last-resumable.pt均保留在runs/<stage>/weights。既有artifacts/final未替換。
困難與解法：見工作紀錄與logs。未解事項：單seed、formal val選模，非獨立test；整套recipe改動，不能歸因於單項超參數。
