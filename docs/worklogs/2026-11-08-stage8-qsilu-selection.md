# 2026-11-08：使用者選定 qSiLU 訓練版 Rep17＋20

## 任務與範圍

使用者先要求選 Rep17＋20，接續明確更正為 qSiLU 版本。依最新指示，採用新輪 `rep17_20` E1，訓練與部署 activation 均為 qSiLU。選擇時間：2026-11-08T15:27:12.713677+08:00。不啟動 Stage9 或新訓練。

## 變更內容與原因

先唯讀核對 optimize 原 Rep 結果、既有 reference-review、本輪 `rep17_20` manifest 與正式 AP。這次為選定既有模型，沒有修改方法、模型權重或訓練配方。

新增 [selection.json](../../yolo_achitechure/result/stage8-retrain/selection.json)，記錄使用者選擇、部署與多分支 EMA SHA、原自動建議、完整十項 AP、算子契約與階段邊界。可攜包 `manifest.json` 的 default_model 改為 rep17_20，新增 user_selection；README 示範改為無參數 load_model() 載入選定模型，並同步 README 的封裝 SHA。原 handoff.json、實驗結果與 source lock 保持原始數據。

同步根層、專項、result、研究索引與計畫、比較報告及 yolo_actshape 接續入口，明確區分「原自動推薦 SiLU 訓練版」與「使用者採用 qSiLU 訓練版」。沒有複製權重、刪除其他候選或增加封存。

## 選定模型

- 部署：rep17_20.pt，SHA `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35`。
- 多分支 EMA：training/rep17_20-best-ema.pt，SHA `7ddd8bebb86dc24a3feab15edd840587dd56e9afeb2a1994f89084687a1afb34`。
- COCO AP 0.514347150、person 0.642795930、COCO 球棒 0.469576310、BBAT box 0.614370693、Pose 0.901389429。
- 訓練 qSiLU／Float PWL；部署為 folded Conv3×3＋qSiLU，完整 D1 integer PWL exp＋LUT256 reciprocal，保留 B2／S1。

## 驗證方式與結果

切換前部署及 EMA checkpoint SHA 核對一致。切換後獨立 CPU 程序呼叫 load_model()（不指定 candidate），實際載入 rep17_20；qSiLU／D1／B2／S1、Rep17／20 位置與 folded 狀態核對通過。160／640 各 24 個輸出 tensors 與既有參考逐值相同，見 [selection-validation.json](../../yolo_achitechure/result/stage8-retrain/selection-validation.json)。410 項研究來源 SHA 未變，原始 handoff.json、README 封裝 SHA、預設入口、相容連結與文件連結均通過。沒有使用 GPU 或重跑正式 AP。

## 困難與解法

無。最初依前文推薦推定為 SiLU 訓練版，使用者更正後立即改以 qSiLU 訓練版為準。

## 未解事項與風險

原自動 AP 保護門檻仍為未通過（球棒 box 相對 parent 降約 0.1401 pp）；使用者選擇不改寫這項實驗事實。單 seed、無等預算 control、未量測硬體加速等限制維持。Stage9 等待使用者指定。
