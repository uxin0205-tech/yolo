# 第 3 階段恢復訓練與 KD 判斷

2026-10-19。使用者要求檢查是否需要 training recovery，以及 KD 是否可加入。原先判斷本階段本身無掉點；使用者隨後明確授權「KD 加在這個 Stage」。**已完成五輪 KD：最後一輪 AP 0.514620791，比 D1 高 0.000431442，未達事前 +0.001 門檻，主線保留 D1。** 詳見 [最終報告](../artifacts/stage3/kd-20261019-v1/RESULTS.md)。

## 完整結果支持的判斷

共同 S1 parent 的官方 COCO AP50–95 為 0.514151658；D1 為 **0.514189349**，差異僅 +0.000037691（+0.00377 個百分點）。D2 為 0.514128963、D3 為 0.514169004，三者皆在事前 0.001 門檻內；不能把微小差距解讀成統計優勢。N4 的 0.514213024 最高，但其當前分母仍為浮點 division，完整 D1 與它只差 0.000023676。

依事前最高 AP 規則選 D1；D2 的 reciprocal 表僅 37 bytes，D1 為 544 bytes，D3 為 136 bytes，均另加共用 exp 表 42 bytes。D2 是有價值的成本選項，保留完整模型與結果；未經硬體綜合不能以 ROM 大小直接判定總面積／延遲。現有 D1 足以交付，沒有理由用訓練掩蓋數值問題或追逐這些微小 AP 差異。

原始模型官方 AP 為 0.524685533，D1 尚差 **0.010496184，即 1.0496 個百分點**。這主要是既有 Binary／scale 主線的累積差距，並非此次 PWL／reciprocal 造成。來源為 [原始 parent 同口徑評估](../../yolo_binaryqk/artifacts/stage1/protocol-audit-20261018/parent-evaluation.json) 與 [本輪比較](../artifacts/stage3/formal-20261019-v1/comparison.csv)。

## 先參考 optimize 的實作與負結果

- [雙教師 KD](../../yolo_optimize/experiments/kd/dual_task_v1/README.md) 已完成五輪：有 Pose 收益，但 ball 框下降，沒有合格 best_joint。
- [Pose-head KD](../../yolo_optimize/experiments/kd/pose_focus_v1/README.md) 最佳 E2 的 Pose AP 0.891329→0.892962；ball 框 0.505192→0.501879、bat 框 0.730823→0.725951，仍未採用。
- [實際 spatial_kd.py](../../yolo_optimize/experiments/kd/dual_task_v1/spatial_kd.py) 使用 layer16／19／22 的通道平均能量圖，L2 normalize 後作空間距離；教師 eval、no_grad、detach，且不進學生 optimizer／EMA／export。這是空間 attention transfer，並非 QK 排名 KD。
- [校準程式](../../yolo_optimize/experiments/kd/dual_task_v1/calibrate_kd.py) 用 train-only batch 的任務與 KD feature 梯度比例決定固定倍率，再確認教師不變。舊 μ、雙任務比例與 LR 不可直接搬到本輪單一 Detect。
- [舊 Attention LR 恢復](../../yolo_attention_final/final/results.json) 最佳相對 audited parent 僅 +0.000194435，未達舊選型門檻；不能因曾有訓練就假定本輪也必須加訓。

外部依據：[知識蒸餾原始研究](https://arxiv.org/abs/1503.02531) 支持訓練時用教師訊號教導單一學生；[Attention Transfer](https://arxiv.org/abs/1612.03928) 支持傳遞空間 attention 資訊。這些只支持方法可行性，不保證本模型恢復 1.05 個百分點。

## 已授權的單一 KD 候選

先以既有原始 `weights/yolo26m.pt` 為教師候選，SHA `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`，學生為本輪 D1。教師不是把已二值適應的學生臨時改成 FP-QK；後者在 optimize 曾驗證更差。原始教師已在相同 evaluator／完整 val5000 評估，但並非每個指標都強於 D1：

| 官方 AP50–95 | 原始教師候選 | D1 學生 | 教師－學生 |
| --- | ---: | ---: | ---: |
| Overall | 0.524686 | 0.514189 | +0.010496 |
| Person | 0.646921 | 0.643273 | +0.003648 |
| Ball | 0.527178 | 0.512681 | +0.014497 |
| Bat | 0.484532 | 0.486021 | −0.001489 |

因此不能直接全圖強制模仿並宣稱教師完全合格。本輪以 train-only 梯度校準與兩處 Q/K 更新檢查為啟動條件，特別保護 bat；採單一經診斷支持的空間 feature KD，先不疊加 QK KL、ranking、logit 或多教師損失。區域限制若需要，明確記為本輪新延伸，不能冒稱 optimize 已驗證。

學生訓練前向必須仍為 B2 Binary Q/K＋S1 Q10／bias＋完整 BitTrue PWL／D1 reciprocal，反向代理另記錄；不能改用 exact exp／除法前向而只在驗證切回。教師與額外 loss 僅在訓練存在，移除後學生推論圖及部署常數不變；訓練會增加教師 forward 與顯存成本。

建議初始上限為一個 KD 候選、最多五輪，從同一 D1 出發；LR、scope、batch 與固定 KD 倍率要在 train-only 梯度／顯存診斷後才固定。本輪不新增未改方法的 control 加訓，故即使有提升，也只能稱「KD 恢復流程的結果」，不能把額外訓練與 KD 的獨立貢獻完全分開。

每輪用同口徑完整 val5000 檢查八項 AP；建議接受條件為 overall 至少 +0.001，八項相對 D1 各不下降超過 0.001，且部署契約不變。不合格維持 D1。這是本輪事前採用門檻；校準、AMP smoke 及正式訓練的實際狀態見工作紀錄，不能提前宣稱有 AP 收益。


## 本輪固定配方與操作

2026-10-19 授權後，先保存 `artifacts/stage3/kd-reference-review.json` 的 optimize SHA 與差異理由，再實作。教師是原始 YOLO26m，學生從 `result/stage3/evaluated.pt` D1 重新載入。採 optimize layer 16／19／22 能量圖 L2 normalization／空間距離，新增 **GT bat 矩形與一格 feature halo 排除**，在 normalization 前遮罩；這是本專案延伸，不能保證共享權重更新不影響 bat，仍需八項 AP 門檻。所有原始標註及 native detection loss 完整保留。

四個真實 train batch（每批 16）以 FP32 feature 梯度校準固定 μ，使 KD／native 梯度中位比例為 0.1，各批須介於 0.025～0.4；不得用 val 選 μ。校準無權重更新，隨後兩個完整 logical macro 做 AMP smoke，驗證 Q/K 真實更新、教師不變、固定係數與 resume。Smoke 的更新丟棄，正式訓練重新從 D1 開始。

AdamW、Attention LR 1e-5／head LR 5e-6、betas 0.948／0.999、weight decay 0.00027、clip 10，延續本輪 S1 配方。所有 BN running statistics、S1 scale／bias、normalizer 常數凍結；只訓練 Attention 與 Detect head。Batch 16、logical 256、native＋KD sum 乘 64／實際 macro 張數；warmup 一輪，再 cosine 至 0.5 倍。完整 train118287，每輪 val5000，最多五輪，單 seed 1，無 control 加訓。

學生訓練使用 integer PWL／D1 reciprocal 前向，Float PWL／exact denominator 僅作反向代理。教師、空間能量 normalization 的浮點除法只存在訓練流程，不加入部署圖。Stage 3 仍保留現有 SiLU，後續 qSiLU 主線在指定 Activation 階段建立。

唯一入口（在 `yolo_attention` 下）：

```bash
PYTHONPATH=src:../yolo_binaryqk ../.venv/bin/python -m yolo_attention.cli stage3-kd init --output artifacts/stage3/kd-20261019-v1
PYTHONPATH=src:../yolo_binaryqk ../.venv/bin/python -m yolo_attention.cli stage3-kd run --output artifacts/stage3/kd-20261019-v1 --execute
PYTHONPATH=src:../yolo_binaryqk ../.venv/bin/python -m yolo_attention.cli stage3-kd status --output artifacts/stage3/kd-20261019-v1
```

單一 QueueExecutor 依序校準、smoke、訓練、checkpoint reload val5000、選型。600 秒背景監測，異常停止自己的 worker；完成後輸出 CSV／SVG／中文報告與 `result/stage3/kd-selection.json`，不啟動 Stage 4。`stage3-kd plot` 只讀小型 JSON，無 GPU／checkpoint 也可重畫。
