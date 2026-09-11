# 2026-09-09：B-HEAD 五 epoch 完成與逐分支更新稽核

## 完成事件

`heads-only-parent-recovery` 由 quiet supervisor 於台北 03:12:34 啟動、03:58:54 正常退出，exit0、status=complete，完成 5 epochs。跨回合持續等待同一 session92210，沒有因觀測 timeout 重啟；正常期間沒有讀 log 或查 GPU。退出後才讀終局 summary。

## 結果與決策

| Epoch | EMA joint | EMA Ball Box | EMA Ball Pose |
| --- | ---: | ---: | ---: |
| Parent | 0.711175 | 0.507437 | 0.859909 |
| E1 | 0.711156 | 0.507163 | 0.859899 |
| E2 | 0.711169 | 0.507234 | 0.860320 |
| E3 | 0.711081 | 0.506118 | 0.860077 |
| E4 | 0.710993 | 0.505681 | 0.860449 |
| E5 | 0.710874 | 0.504551 | 0.860347 |

E5 相對原 native5，Ball Box 0.502376→0.504551、joint 0.710116→0.710874，表示限制 Neck 更新減輕退化；但仍低於原 BEST，Ball Box delta=−0.00288633，超過驗收容許退化0.001。E2 的 joint 也沒有超越 parent。沒有新可接受 recipe，不替換原 BEST，不直接加訓這一支。

Live Ball Box 五輪均低於 parent，E5 delta=−0.00896035。此 run 沿既有 native 對照：EMA 決定選模／安全暫停，live 只診斷，沒有開啟 `pause_on_live_regression`；不能把 complete 說成 live 八項都通過。尚未量到可信的視覺收益足以代替增準門檻。

## 排除「需要的 head 子分支沒在學」

新 `scripts/audit_head_updates.py` 在 CPU 讀取五個完整快照，逐 epoch 核對 585 個固定 state 在 live／EMA 都與 parent 完全相等。依 optimizer param_names 對齊每個子分支，稽核實際參數變動、AdamW exp_avg 與 step。

E5 的 Detect／Pose box、classification、Pose feature、kpts、sigma、flow 全部有非零 moments 及參數變動；各 active tensor 的 step=2315。Pose one-to-one 點位輸出6個 tensors 全部有變動；不是只更新 classifier 卻漏掉 keypoint head。moment 是累積證據，不代表每個 minibatch 的梯度都非零。

E5 Pose one-to-one relative L2：box 約2.59%、classification1.05%、Pose feature3.71%、kpts0.816%、sigma0.321%。這些不同尺度的變動不直接等同不穩定，也不單憑大小改 LR。

產物：`artifacts/direction1-20260908/heads-only-parent-recovery/summary.json`、`head-update-audit.json`，全部 checkpoint 保留。CPU audit exit0、五輪固定 state 檢查通過；沒有新 GPU 推論、資料或權重變更。

## 待續

完整優化目標仍未達成。目前已排除空梯度／scope 失效，不把安全保存誤認為增準。下一步檢查這個固定共享特徵模型的 head BN 訓練／推論差異，再依證據決定是否需要 BN 策略對照；不直接把先前不同 run 的 BN-only 結果當成這次的根因。QK 合法梯度／teacher 與長訓 optimizer 候選仍依原計畫前置條件，未擅自啟用。

困難：無。路徑實際為 `checkpoints/` 而非 `full-resume/`，經唯讀列表確認後使用正確來源，未建立替代快照。沒有刪除、commit、push、改 split／labels 或使用子代理。
