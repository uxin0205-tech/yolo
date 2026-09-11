# 新舊 combine 的 ball／bat 差距與恢復結果

更新：2026-09-11。這是本輪差距診斷與最小恢復試驗的結果，不是整個融合／activation／方向 2 已完成。

## 結論

差距真實存在；不是用錯 BBAT split、PWL 範圍不同或舊分數無法重現。新模型保住更多 COCO 精度，但目前未兼顧舊 combine 的 BBAT 表現。
固定共享層／Detect，只補訓 Pose head，能追回部分 ball／bat 框 AP；尚未追回全部，也沒有全面提升關鍵點。不能把本輪 `best_pose.pt` 稱為通過驗收的 `best_joint.pt`。

## 完整同口徑比較

全數為 AP50–95，非只有 AP50。COCO val 5,000 張、canonical BBAT5 v1 val 683 張，imgsz 640，Ultralytics 8.4.90，正式 Bit-True，PWL [-10,0]／20 段。ball／bat 框由二類 Pose 模型的 box 分支評估，不是將 COCO80 類別直接當成 BBAT 二類。

| 指標 | 舊 combine J3 best_joint | 新 bridge J3 E12 | Pose head 補訓 E5 | 補訓減起點 |
| --- | ---: | ---: | ---: | ---: |
| COCO box | 0.498022 | 0.504242 | 0.504242 | 0 |
| person box | 0.620381 | 0.626983 | 0.626983 | 0 |
| BBAT box | 0.630036 | 0.603892 | 0.608773 | +0.004881 |
| BBAT pose | 0.903717 | 0.886071 | 0.886747 | +0.000676 |
| ball box | 0.507437 | 0.486159 | 0.492201 | +0.006042 |
| ball pose | 0.859909 | 0.855779 | 0.854418 | -0.001361 |
| bat box | 0.752634 | 0.721624 | 0.725345 | +0.003721 |
| bat pose | 0.947526 | 0.916362 | 0.919075 | +0.002713 |

補訓後比舊 combine 的 ball box 仍低 0.015236、bat box 低 0.027289、bat pose 低 0.028450。不能只展示改善幅度而省略剩餘差距。

## 哪些原因已經驗證？

### 1. 不是 PWL 範圍或本次 Bit-True 數值差異

新舊實際模型都通過 [-10,0]、20 段、固定非訓練表格檢查。新模型 Float 與 Bit-True 八項 AP 最大差約 0.00013，比目前 BBAT 差距小很多。
注意：兩種 backend 都使用 BinaryQK；這個比較不等於 BinaryQK 與全浮點 Q/K 的比較，不能據此宣稱 BinaryQK 沒有精度代價。

### 2. 新 Pose 的 MASF 推論開關不會直接改變結果

```text
layer16：p3_raw ─┬─> layer17 → P4 → layer20 → P5
                ├─> MASF(α) → Detect P3
                └───────────> Pose P3
layer19：p4_raw ─────────────> Detect／Pose P4
layer22：p5_raw ─────────────> Detect／Pose P5
```

新 α=0.01712549；同權重 α=0 後六項 BBAT AP 完全不變。這是路徑決定的結果，不是只因 α 小。
舊 α=0.11065911，MASF 在共享 layer16。舊同權重 on−off：

| 舊 MASF 的推論變化 | on−off AP |
| --- | ---: |
| ball box | +0.000922 |
| ball pose | +0.004465 |
| bat box | -0.000089 |
| bat pose | -0.000359 |

因此舊 MASF 的當下推論貢獻不足以解釋約 0.031 的 bat 優勢。但不能排除 MASF 在舊訓練歷史中塑造共享特徵的作用；推論消融不是無 MASF 重訓的因果對照。

### 3. Pose head 的適應確實是部分因素

補訓只更新 Pose head；匯出時確認 827 個非 Pose 張量與 J3 起點逐位相同，360 個 Pose 張量改變。COCO／person 全量 AP 完全不變，ball box +0.006042、bat box +0.003721。因此不改 backbone 也能追回一部分差距。
但補訓在第 9 epoch 平台停止，最佳為 E5；ball pose 略降，說明增加相同 head-only 訓練不是全面解法。

### 4. Pose head BN 統計不是主要解方

使用完整 5,964 張 canonical train，關閉隨機增強，只重估 48 層 Pose head BN 的 running statistics，所有參數／其他 buffers 不變。physical batch 128、峰值 allocated memory 8,872,495,104 bytes（約 8.87 GB／8.26 GiB）。
相對補訓最佳：BBAT box +0.000196、Pose +0.000516、ball box +0.000444、ball pose +0.000366、bat box -0.000052、bat pose +0.000666。量級太小且非全面提升，保留為診斷候選，不升格。
這只是無反向的校準，不是證明完整訓練也能使用 physical batch 128。

## 訓練與選模設定

J3 完成 17 epoch／7,871 macro，patience 5。恢復試驗另開 run：AdamW、Pose head LR 2e-5、warmup 1、cosine final factor 0.5、最多 10 epoch、patience 4、physical Pose batch 16、weight decay 0.00027、betas (0.948,0.999)、原 seed 20261003、AMP、clip norm 10；無新增資料或增強方案。
恢復試驗實際完成 9 epoch，依 mean(box AP,pose AP) 選 E5，未因回合數到達而自動採用。每輪 full COCO／BBAT 評估，固定 Detect AP 必須完全不變；單項 BBAT 比起點下降超過 0.02 即保存停止。

正式 gate 仍比最初的獨立 Detect／Pose：COCO 降幅不超過 0.005、六項 BBAT 不超過 0.02。恢復後 ball box 已過此門檻，其餘五項 BBAT 尚未全部恢復；沒有 best_joint。

## 下一步優先順序（尚未執行）

1. 優先驗證共享特徵適應限制：與舊 combine 相比，目前共享 BN affine 固定、LR 較低、任務梯度權重不同；完整獨立 Pose 階段本身也未達舊 Pose 精度。以相同起點做受控的共享 affine／Neck 適應，保留 COCO 原門檻。只改一組可訓練範圍，先校準實際更新，再決定長訓。
2. 若共享微調仍形成 COCO／BBAT 取捨，才考慮訓練期的任務保護／蒸餾；必須另立清楚對照，不能把它混入目前最小恢復試驗後宣稱單一因素有效。
3. 不直接加大相同 head-only epoch、不因舊 paper 較好就移動 MASF、不放寬門檻換取通過。activation／方向 2 尚未開始，亦沒有自動 queue。

共享特徵限制目前是下一個合理假設，不是已由單變因試驗證實的根因。現有多項訓練差異不能全歸因於 AdamW 或 MuSGD 任一 optimizer。

## 權重與證據位置

- 恢復候選：`artifacts/fusion/j3-pose-head-recovery-v1/inference/best_pose.pt`，SHA256 `d5b2083b2b736ded790578a2779d223d82413ce0c6cbac1468601105c094f9a6`。僅 inference state，不是 exact resume；完整續訓檔在同 run 的 `checkpoints/best_pose.pt`。
- 新舊完整重驗與消融：`artifacts/old-combine-comparison-v1/summary.json`。
- J3 原結果：`artifacts/fusion/balanced-j3-v1/summary.json`。
- 補訓停止結果：`artifacts/fusion/j3-pose-head-recovery-v1/recovery-stop.json`。
- BN 診斷：`artifacts/j3-pose-bn-calibration-v1/summary.json`，candidate.pt 未升格。
- [中文工作紀錄](<../../docs/worklogs/2026-09-11-j3-result-bbat-diagnosis.md>)。

## 困難與限制

新建舊 Detect 診斷骨架一度漏設非 state-dict 的 BN eps，修正為 YOLO 的 1e-3 後舊八項 AP 完全重現；只重跑失敗的舊案例，保留所有輸出。原訓練模型未受此新建診斷工具問題影響。
其他執行困難：無。尚未驗證第二 seed、測試集泛化、實際硬體 latency 或進一步共享層補訓。所有 AP 增益都是目前固定 validation 上的觀察。
