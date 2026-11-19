# YOLO26 Detect–Pose 融合設計規格

- 日期：2026-08-21
- 更新：2026-08-22
- 狀態：設計共識與來源驗收已確認，進入 Full35 主線實作
- 範圍：軟體模型、訓練、驗證與推論介面；FPGA 實作不在本階段範圍

## 1. 問題與目標

目前流程以兩個獨立 YOLO 模型處理不同任務：

- Detect：偵測人，結果交給外部 ViTPose 流程。
- Pose：偵測球與球棒，並輸出對應 keypoints。

本專案要將兩個任務收斂到同一套可訓練、可比較、可逐步融合的 YOLO26m 工程中。第一個真正的融合候選會讓 Detect 與 Pose 共用一次 Backbone + Neck 特徵抽取，再接兩個獨立 head。設計同時保留雙模型封裝作為低風險基準與失敗回退方案。

### 主要目標

1. 建立可重現的獨立 YOLO26 Detect/Pose 基準。
2. 提供統一的雙任務推論介面，不再讓應用程式直接管理兩個模型。
3. 實作真正共用權重的 Backbone + Neck 與 Detect/Pose 雙 head。
4. 儘量維持兩個獨立模型的精度，同時降低合計參數、MAC 與 YOLO 推論延遲。
5. 保留後續導入既有 attention、PWL normalizer、MASF 與量化方法的明確接點。

### 已選定的架構來源

- 權威來源：`/home/uxin/yolo/yolo_achitechure/achitechure_1/final/`。
- 主線：`Full35-A2`，訓練初始化使用 Float checkpoint，正式比較另保留 Bit-True checkpoint。
- 備案：`Partial75-A2`。
- 被來源 bundle 標記為 `rollback` 的 B/C checkpoint 不作正式主線初始化。

### 本階段非目標

- 不修改或合併 ViTPose；它仍是 Detect person 結果的外部消費者。
- 不先做 FPGA/HLS/上板實作，也不宣稱硬體效益。
- 不以權重平均作為模型融合方法。
- 不在第一版實作統一 detection/keypoint head。
- 不使用追蹤器或隔幀任務排程來改變目前逐幀完整輸出的語意。

## 2. 任務與資料語意

### Detect 任務

- 主要應用輸出：`person` bounding box。
- 第一個共享模型候選使用完整 COCO2017 80 類 supervision（F1-80），應用層只消費 person。
- 後續另做 person-only、`nc=1` 的精簡候選（F1-1）。它必須使用獨立衍生且可稽核的 person-only data view，不能用會把其他類別誤映射成 person 的 `single_cls` 捷徑。

### Pose 任務

- 類別：`ball`、`bat`。
- 共同 schema：`kpt_shape=[2,3]`。
- ball：第一點有效，第二點由 visibility mask 排除。
- bat：保留兩點；在資料語意尚未由人工確認前，只命名為 `endpoint_0` 與 `endpoint_1`，不推定棒頭或棒尾。
- ball 輸出一律來自 Pose head；不保留舊主程式中以 COCO sports-ball box center 代替 ball keypoint 的模式。

### Joint GT

Joint ground truth 指同一張影像同時具有：

- person bounding box；
- ball/bat bounding box；
- ball/bat keypoints。

它用來驗證融合模型在同一場景的整體行為，不是把兩個資料集拼在一起的別名。現有 BBT 中有一部分影像可與 COCO 真實 person 標註對應，能建立小型 joint subset；正式 finalist 前必須完成其隔離與人工抽查，但不阻塞最初 smoke test。

## 3. 模型演進階段

### F0：兩個獨立 YOLO26 基準

- D0：YOLO26m Detect。
- P0：YOLO26m Pose，負責 ball/bat。
- 兩者各自有完整 Backbone、Neck、head 與權重。
- 目的：建立精度、參數、MAC、VRAM 與延遲上限，也是所有融合結果的對照組。

### F0.5：RoutedDualModel

將 D0 與 P0 放進同一 Python API 與 checkpoint bundle，由 `tasks` 參數路由：

```python
result = model(frame, tasks={"detect", "pose"})
```

特性：

- `detect`、`pose`、`both` 三種執行模式。
- 仍有兩套完整 trunk 權重；`both` 仍需跑兩次 trunk。
- 預期接近兩個獨立模型的精度。
- 只解決整合、封裝與介面一致性，不宣稱降低演算法 MAC、權重容量或總延遲。
- 同時作為 F1 精度不合格時的正式回退方案。

### F1：Shared Backbone + Neck，雙 head

```text
Input
  └─ Shared YOLO26m Backbone + Neck
       ├─ Detect head → COCO classes；應用消費 person
       └─ Pose head   → ball / bat boxes + keypoints
```

核心條件：

- Backbone 與 Neck 的 module parameters 必須是同一組物件與同一組權重，不能只是兩個相同結構的模型。
- P3/P4/P5 只計算一次，再送往兩個 head。
- Detect 與 Pose head 各自保有獨立參數與 loss。
- 同一張影像需要兩種結果時，兩個 head 都會執行；輸出端 MUX 不會被描述成省去已發生的計算。

候選順序：

1. **F1-80**：Detect head `nc=80`，用完整 COCO supervision。這是第一個實作候選，降低資料處理變因並保留完整 Detect 知識。
2. **F1-1**：Detect head `nc=1`，只偵測 person。若它通過精度門檻，因 head 更小而成為優先精簡候選；否則保留 F1-80。

### F2：Unified detection branch + conditional keypoint branch

F2 讓單一 box/class branch 同時處理 person、ball、bat，再只對 ball/bat 啟用 keypoint branch。它可能進一步減少重複 head 計算，但需要自訂 class-conditioned keypoint loss、label masking、trainer 與 export contract。

F2 是 F1 成功後的研究候選，不是第一版交付內容。

## 4. 推論與輸出契約

F0.5 與 F1 必須共用同一呼叫介面，使應用層不依賴模型內部是否真的共享 trunk。

```python
result = model(frame, tasks={"detect", "pose"})
```

最低輸出物件：

- `persons`：person boxes、confidence 與 class metadata。
- `balls`：ball boxes、confidence 與可見 keypoints。
- `bats`：bat boxes、confidence 與 `endpoint_0`/`endpoint_1`。

對既有影片 JSON 的相容層維持：

```json
{
  "frame_index": 0,
  "persons": [],
  "balls": [],
  "bats": []
}
```

允許以不破壞既有 consumer 的方式增加模型版本、checkpoint ID、task source 等 metadata。person crop 與 ViTPose 呼叫由 adapter 或既有 pipeline 負責，不放入融合模型核心。

第一個軟體版本逐幀回傳完整 person、ball、bat 結果。非同步排程、CUDA stream 或 Detect 完成後提早啟動 ViTPose，可在介面穩定後另行優化。

## 5. 資料策略

### Detect

- F0 與 F1-80 使用現有完整 COCO2017 train/val，不先切掉其他類別。
- 評估至少單獨報告 person box mAP50-95；完整 80 類結果可作防遺忘參考。
- F1-1 才建立 person-only 衍生資料集：原圖可用 symlink，label 只保留 person，保留沒有 person 的影像作真實負樣本，並保存 manifest、計數與 hash。
- COCO sports ball/baseball bat 不作第一版 Pose keypoint supervision；日後可作 auxiliary box ablation，但不能被當成 keypoint GT。

### Pose

- 原始來源：`/home/uxin/yolo/original/pose/dataset/`。
- 原始資料保持唯讀。
- 正式 P0/F1 採 leakage-safe 的衍生 Pose view：以既有 locked assignment 為基礎，再做 session/source-aware 檢查，保留 11 欄 Pose labels，修正已知極小負座標並產生 manifest。
- 現有 grouped Pose split 可用於快速 smoke，但不能直接宣稱是最終無洩漏基準。
- 同一來源影像的亮度 augmentation siblings 不得跨 split，訓練 sampler 也應避免同一 optimizer update 重複抽到同 source siblings。

### Joint subset

- 從 BBT 與 COCO 可確定對應的原始影像建立真實 joint subset。
- joint validation 的 source 必須從 Detect train 移除，避免共享 trunk 在另一任務訓練時看過相同底圖。
- validation 每個原始 source 原則上只保留一個代表影像，避免把亮度變體當成獨立樣本放大樣本數。
- finalist 前人工抽查標註完整性；另可從獨立 sample video 稀疏抽幀人工標註，作 domain case study，而不能把既有模型預測 JSON 當成 GT。

## 6. F1 初始化與解凍策略

第一版不平均 D0 與 P0 的 trunk 權重。

1. 以已驗收的 Full35-A2 Float Detect/attention/MASF 權重初始化 shared trunk。
2. Detect head 載入相容的 D0 head 權重。
3. Pose head 載入相容的 P0 head 權重。
4. 初期凍結 shared trunk，只讓 heads 適應共同 graph。
5. 先解凍 Neck，再視 validation 與 gradient 診斷逐步解凍 Backbone。
6. 全程使用同一 optimizer 管理 shared parameters 與兩個 heads，避免同一 shared parameter 被兩個 optimizer 以不同狀態重複更新。

若來源 checkpoint 無法安全映射，必須停在 Source Acceptance Gate 回報，不得靠 layer index 猜測性複製。

## 7. 多資料集訓練與 loss routing

COCO 與 BBT 是 disjoint annotation datasets，因此每個 batch 只能對它實際擁有的標註計算 loss：

- COCO batch：計算 Detect loss，更新 Detect head 與 shared trunk；不計算 Pose loss。
- BBT batch：計算 Pose box/class/keypoint loss，更新 Pose head 與 shared trunk；不把未標註 person 當背景。

第一輪比較兩種 task ratio：

- R1：Detect:Pose = 1:1。
- R2：Detect:Pose = 2:1。

2:1 的一個 optimizer update 定義為兩個 Detect microbatches 與一個 Pose microbatch。為避免 Detect 僅因 microbatch 數量而自動加倍權重，先平均兩個 Detect loss：

```text
L_det = (L_det_1 + L_det_2) / 2
L_total = lambda_det * L_det + lambda_pose * L_pose
```

初始 `lambda_det=lambda_pose=1`。訓練進度以 optimizer steps 與每個 task 看過的有效樣本數報告，不只寫 epoch。

2:1 可以作為候選，但資料量較大不代表它必然更好。R1/R2 由以下證據選擇：

- 各任務 validation 曲線與過擬合時間點；
- shared trunk 的 task gradient norm；
- Detect/Pose gradient cosine similarity；
- 每個任務相對獨立 baseline 的退化量。

固定 loss baseline 若已符合精度門檻就不增加演算法。只有在觀察到明顯 gradient imbalance/conflict 時，才依次評估 loss weight 調整、GradNorm 或 PCGrad，且每項皆需獨立 ablation。

## 8. 驗證、比較與驗收

### 主要精度指標

所有門檻以 mAP50-95 的絕對差值計算，且逐項套用：

- person box；
- ball box；
- bat box；
- ball pose；
- bat pose。

對每個指標：

```text
fused_metric >= corresponding_independent_baseline_metric - 0.08
```

例如 baseline 為 0.80，融合後不得低於 0.72。任何一項失敗都不能用其他任務的平均提升抵銷。

### 效率指標

模型層級至少報告：

- 參數量與 checkpoint 大小；
- MAC/FLOPs，固定輸入尺寸與計算工具；
- peak VRAM；
- batch=1 latency，包含 warmup、同步與分位數。

Pipeline 層級另外拆分並報告：

- shared/independent YOLO；
- CPU 後處理與同步；
- person crop；
- ViTPose；
- 每幀總延遲。

F1 除精度門檻外，還應滿足：

- 模型參數與 MAC 低於 D0 + P0 合計；
- `tasks={"detect","pose"}` 的 YOLO batch=1 latency 低於兩個獨立模型串行執行。

若環境噪聲使延遲無法穩定區分，須報告量測分布，不能只以單次平均值宣稱勝出。

### Seeds

- 開發與 smoke：1 seed。
- finalist：暫定 2 seeds。
- 若時程只允許 1 seed，結果必須明確標為 provisional，不宣稱已建立跨 seed 穩定性。

## 9. Source Acceptance Gate

使用者之後提供的整理完成架構資料夾，是 attention、PWL、MASF、權重與訓練實作的權威來源。它保持唯讀，融合工程實作於本專案中。

正式設計 module mapping 與 trainer 前，來源必須依序通過：

1. 環境、版本、license 與 checkpoint provenance 可辨識。
2. Detect 與 Pose checkpoint 可載入，輸入/輸出 schema 明確。
3. 單任務 forward 與最小 smoke validation 可重現。
4. graph audit 確認 shared trunk 邊界、P3/P4/P5 接點與 attention/MASF 實際位置。
5. state-dict mapping 可由 module name、shape 與語意驗證，不依賴脆弱的裸 layer index 假設。
6. 最小 smoke training 能 backward、optimizer step、save 與 reload。
7. bit-true/PWL/量化功能的軟體 reference 與限制分開記錄；不把 fake quant 或仍含 float reciprocal 的路徑描述成完整 FPGA datapath。

通過後才產出精確到檔案、class、method 與測試案例的 implementation plan。

## 10. 失敗回退與擴充順序

若 F1 未達精度門檻，依最小複雜度順序檢查：

1. 確認資料 routing、label masking、head 初始化與 metric 對齊沒有錯誤。
2. 調整 task ratio、freeze/unfreeze 時程與 loss weights。
3. 加入 task-specific BatchNorm 統計或 affine parameters。
4. 在 Neck 後段做 partial split。
5. 加入小型 task adapters。
6. 若仍不合格，正式交付 F0.5，保留雙模型精度與統一 API。

attention、PWL 與 MASF 會從最終來源架構繼承，但仍需做 ablation：原生 YOLO26、attention 版本、attention+MASF 版本不得混成同一個無法歸因的實驗。量化則在 float F1 通過精度門檻後進行。

## 11. 主要風險

- **Negative transfer**：COCO person 與棒球 Pose domain/樣本量差距可能讓 shared trunk 偏向 Detect。
- **Normalization domain shift**：共用 BN 統計可能混合兩個資料域，必要時採 task-specific BN。
- **Partial-label false background**：對缺少的任務標註計算 loss 會直接破壞訓練，必須由 task-aware routing 阻止。
- **資料洩漏**：BBT augmentation siblings、影片相鄰幀及 BBT/COCO 重疊來源都必須 group-aware 隔離。
- **舊權重世代差異**：YOLO11m Pose 權重只能作 baseline/teacher，不能假設可直接拼接到 YOLO26m。
- **自訂算子可移植性**：attention/PWL/MASF 的 export、梯度、量化與 latency 必須各自驗證。
- **授權**：目前資料與 Ultralytics checkpoint 的授權需在研究外部署前重新確認；本階段只定位為研究原型。

## 12. 實作啟動條件與交付物

收到最終架構來源資料夾並通過 Source Acceptance Gate 後，下一份 implementation plan 應固定：

- package 與檔案結構；
- shared trunk/head 的 module 邊界；
- F0.5/F1 forward 與輸出 adapter；
- dual-loader trainer、loss routing 與 optimizer step；
- dataset derivation、manifest 與 leakage tests；
- single-task、joint、latency validators；
- checkpoint save/load/export contract；
- 單元、整合、smoke 與回歸測試；
- F0 → F0.5 → F1-80 → F1-1 的可重現執行順序。

在來源尚未到位前，本文件是架構與實驗契約；不先猜測最終 class 名稱、layer index 或 state-dict key。
