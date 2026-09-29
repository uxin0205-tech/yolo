# Detect 大教師與 BBAT5 Pose 教師研究

2026-09-11。此為教師候選與方法修訂建議，尚未下載大模型或啟動 KD 訓練；原單 FP-QK joint teacher 規格保留。

## 結論與官方依據

Detect 可評估 YOLO26L；更大的 X 也只是候選，必須先用本專案相同 COCO val、解析度、後處理確認 overall／person 優势，不能把官方表格直接當本機結果。[官方 YOLO26 模型](https://docs.ultralytics.com/models/yolo26/) 列有 Detect 與 Pose 的 L／X 尺寸。

官方 `yolo26l-pose.pt` 的預訓練任務是人體 17 點。[官方 Pose 文件](https://docs.ultralytics.com/tasks/pose/) 支持自訂資料訓練。本機 canonical `configs/pose.yaml` 為 ball／bat 兩類、`kpt_shape: [2, 3]`；因此人體權重只能作初始化候選，不能直接把其人體關鍵點輸出當 BBAT5 教師標的。若要大 Pose 教師，必須先按 canonical BBAT5 訓練適配並驗證，不能更改 split。

[官方 KD 文件](https://docs.ultralytics.com/guides/knowledge-distillation/) 提供 Neck 特徵投影對齊不同通道的蒸餾做法，並明確指出目前只有 Detect 的精度收益經實驗驗證，Pose 技術相容不代表已證明收益。官方範例不是本專案自訂雙 head／BinaryQK trainer 的直接相容性保證，不能直接套 `distill_model` 即宣稱完成。

## 建議的最少實驗

1. Detect：驗證 YOLO26L 的 COCO overall／person；有明確教師優勢後才進蒸餾，不先耗費資源重訓大 Detect。
2. Pose：優先使用已訓好的原獨立 BBAT5 Pose，核對資料 lineage 與同口徑結果。教師可以與學生同尺寸；本專案已有其 BBAT pose AP 0.912161、bat pose 0.948058，高於新 qSiLU 學生 0.891329／0.923008 的證據。高 AP 只代表有希望，仍需檢查錯誤重疊與蒸餾實驗，不保證收益。
3. 若確認修訂為雙教師，COCO batch 只用 Detect 教師，BBAT5 batch 只用 Pose 教師；同圖、同增強、同任務 class／keypoint 語義。保留真值 native loss，不補造另一任務的標註。
4. 跨尺寸模型的 attention head 數量、通道及位置未必對應。先設計可對齊的單一 KD 訊號與 K0 對照，不把原單教師排名 loss、額外 feature loss、輸出 loss 全部堆入一次實驗。區域創新需等普通 KD 有效，另與同預算對照比較。
5. 舊 Pose 教師沒有足夠有效訊號，才考慮另外訓練 YOLO26L-Pose 的 BBAT5 版本；不預設需要額外大教師訓練。

教師與可能的特徵投影僅存在訓練期，正式匯出仍為既有學生；不因此新增部署 head、逐圖 scale 或更大 backbone。GPU 訓練成本會增加，推論成本是否維持需實際匯出驗證。

## 已完成的安全前置

`artifacts/score-gradient-preflight-v1.json`：兩個 attention site 的原生 binary score 沒有 Q/K 輸入梯度；既有 training-only surrogate 前向完全相同，Q/K 梯度有限且非零，eval 與 state 不變。本結果只驗證 score 元件，不等於全模型 KD 更新、AMP、資料映射已通過。

新學生與全量教師切換反證見 [Activation 結果](<../../activation/bridge_v1/RESULTS.md>)。直接 FP-QK 切換的候選八項全降，不採用。雙教師研究涉及原 R2-REGION 的教師與 loss 邊界修訂，尚待確認，不靜默替換原方案。
