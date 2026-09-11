# 2026-09-11：Pose head 專項啟動準備與共享特徵授權

## GPU 前置完成

calibrate／native smoke／KD smoke 全部通過。μ=10.0635533732，對應 head feature 梯度約原生5%，不是loss本身乘10就代表強蒸餾。四個head分支均有live KD梯度；teacher state／無梯度、非Pose state及Detect同圖輸出精確不變通過。峰值allocated：native5,191,114,240／KD5,527,556,096 bytes。資料未重切，probe更新不作正式權重。下一步正式兩臂各5輪，shared小範圍實驗仍待結果決定起點。

## 變更與原因

使用者核准執行前一輪Pose-only規劃，並允許安排共享特徵試驗。新增 `kd/pose_focus_v1/`，實作獨立設定、四組Pose26 head隱藏特徵KD、train-only倍率校準、兩macro smoke與兩臂正式queue。原教師／activation／dual-task結果均保留，不改原trainer或checkpoint。

## 驗證方式與目前結果

CPU實際configure、import與AST通過：stages僅j0，physical16、shared_bn_affine_trainable=False，只有pose_head LR1e-5，epochs5／warmup1。teacher safe_source仍限定hash與weights_only。新建state_checks避免同名experiment模組跨研究分支誤用。GPU前置將查核teacher state、四head組KD梯度、非Pose state及Detect同圖輸出精確不變；尚不能稱正式訓練完成。

## 共享特徵接續

已記錄第二階段S0／S1小範圍P3輸出Conv解凍對照設計，先等head-only結果，再鎖定實際module／LR／起點。這不是僅關閉person loss就保證不退步；共享變動時仍完整COCO/person驗證。未預先啟動共享層job或新增部署adapter。

## 困難與解法

讀取歷史j0_runtime路徑時發現其位於combine根層，已使用正確路徑；無GPU訓練失敗。現有檔案編輯遇sandbox bwrap限制，使用限定補丁的提升權限apply_patch。其餘無。正式結果、梯度校準數值、共享特徵效果仍待驗證；不承諾必然增準。GPU工作600秒事件監測，正常不讀log。
