# 保留Pose精度的框定位精修結果

相同逐階段降LR P3 best初始化、固定640／batch32訓練；對照Box loss7.5，候選10.0，其他參數相同。
固定bbat5-v1 formal val 683張、評估640／batch8；兩個參考模型引用既有同口徑結果。

| 模型 | Box AP | Pose AP | Ball Box | Ball Pose | Bat Box | Bat Pose |
| --- | --- | --- | --- | --- | --- | --- |
| combine_recipe_p3_640 | 0.634452 | 0.917011 | 0.539151 | 0.894054 | 0.729754 | 0.939968 |
| stagewise_lr_p3_640 | 0.628808 | 0.920616 | 0.529158 | 0.896941 | 0.728458 | 0.944291 |
| box_recovery_control_640 | 0.628824 | 0.919603 | 0.531048 | 0.895997 | 0.726600 | 0.943210 |
| box_recovery_box10_640 | 0.628729 | 0.920083 | 0.528250 | 0.896178 | 0.729209 | 0.943988 |

條件式建議：stagewise_lr_p3_640。

條件：相對本次保留模型平均AP提高且Box AP至少提升0.001；整體及每類Pose AP下降不超過0.003；每類Box AP下降不超過0.005。這些是工程容許值，不是統計顯著門檻。
不自動替換任何舊權重。各run保留best.pt、last.pt、last-resumable.pt；診斷summary另保存固定conf0.25的漏檢與像素誤差。
困難與解法見工作紀錄。限制：單seed、formal val反覆選模；Control vs Box10可比較loss加權，但不能只以加權後train loss下降宣稱精度提高。
