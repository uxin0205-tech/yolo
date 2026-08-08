# BBT5 是否需要大 receptive field：CPU 實驗計畫

## 1. 研究問題與範圍

本計畫要回答：

> BBT5 的 `ball` 偵測是否需要球本身以外的大範圍上下文？效能在 1×、2×、4×、8× bbox context 或全圖的哪一點飽和？

不能只靠肉眼、理論 receptive field 或 MASF-YOLO 的結果回答。小球在 feature map 上很小，不等於它一定需要大 context；它也可能只需要較高解析度的局部特徵。

本輪只做兩個核心實驗：

1. **Scale-to-cell statistics**：球在 P2/P3/P4 還剩多少 cells。
2. **Context Radius Experiment**：逐步移除球周圍以外的資訊，量測 AP(r) 與 Recall(r)。

全部使用 `../bbt5-detect-baseline`、既有 detect checkpoint 與 CPU inference。不重訓、不驗證 MFAM、不修改模型架構。

## 2. 先定義「需要大 receptive field」

本計畫不以 theoretical receptive field 的像素公式作結論，而以模型行為定義：

- 若 2× 或 4× bbox context 已與全圖相當，表示額外遠距上下文收益有限。
- 若 8× 仍明顯優於 4×，表示模型至少需要較大的局部場景。
- 若全圖仍明顯優於 8×，才有證據支持遠距／全場 context 有實際價值。

這裡的結論是「需要多少輸入上下文」，不是直接判定某個 layer、branch 或 channel 可刪除。要定位到 layer／branch／channel，仍需另一輪模型內部 ablation；本輪先判斷有沒有做該輪實驗的必要。

## 3. 實驗一：球在各尺度剩多少 cells

### 3.1 資料與 resize 規則

分別統計 train 與 valid 的 class 0 (`ball`)。必須模擬 Ultralytics 在 `imgsz=640` 的 letterbox，而不是直接把 normalized bbox 乘 640。

對原圖寬高 `W, H`：

```text
scale = min(640 / W, 640 / H)
w_px = bbox_w_normalized × W × scale
h_px = bbox_h_normalized × H × scale
```

對 stride `s`：

```text
w_cell(s) = w_px / s
h_cell(s) = h_px / s
```

至少計算：

| Feature | Stride |
| --- | ---: |
| P2 | 4 |
| P3 | 8 |
| P4 | 16 |

### 3.2 必須輸出的統計

每個 split 輸出：

- ball bbox width histogram。
- ball bbox height histogram。
- ball bbox area histogram，建議 x-axis 使用 log scale。
- P2/P3/P4 的 `w_cell` 與 `h_cell` histogram。
- `min(w_cell, h_cell)` 的 cumulative distribution。
- 每層 `<1 cell`、`1–2 cells`、`2–4 cells`、`>=4 cells` 的比例。
- p10、p25、median、p75、p90，不只報平均值。

### 3.3 初步量測結果

以現有資料在 640 letterbox 後計算：

| split | ball 數 | bbox 中位數 | 面積 `<32² px` | 中位最短邊 P2/P3/P4 |
| --- | ---: | ---: | ---: | --- |
| train | 3,312 | 16×16 px | 78.4% | 3.75 / 1.88 / 0.94 cells |
| valid | 301 | 16×17 px | 82.4% | 3.75 / 1.88 / 0.94 cells |

目前可說：典型球在 P4 已不到 1 cell；P3 約 2 cells，偏小但並非完全無法表示。這支持繼續做 context 實驗，不能單獨證明需要 P2 或大 receptive field。

### 3.4 此實驗的判讀

- 若多數 ball 在 P3 的寬或高 `<1 cell`：有直接資料證據支持 P2／更高解析度特徵。
- 若多數在 P3 仍有 `>=2 cells`：P3 至少具有基本空間表示能力，是否需要大 context 必須由實驗二判定。
- cell 數只回答 spatial resolution，不回答遠距場景是否重要。

## 4. 實驗二：Context Radius Experiment

### 4.1 Context 定義

對每個 ground-truth ball，以 bbox center 為中心，寬高等比例放大：

| 條件 | 可見區域 |
| --- | --- |
| R1 | 1× bbox，主要只有球 |
| R2 | 2× bbox，球與緊鄰背景 |
| R4 | 4× bbox，可能包含球棒、手或人體局部 |
| R8 | 8× bbox，更大的局部場景 |
| FULL | 完整原圖 |

若窗口超出影像邊界，直接 clip 到影像範圍。球的位置、bbox 大小及整張輸入尺寸維持不變。

另外記錄每個相對窗口在 640 letterbox 後的實際像素寬高。例如中位 ball 約 16×17 px，R8 約為 128×136 px。這讓結果仍能轉換成實際 context 距離。

### 4.2 Primary intervention：遮罩，不改變球尺寸

對 context 外部做兩種處理：

1. **GRAY**：替換成固定中灰色，例如 RGB `(114,114,114)`，與常見 letterbox padding 接近。
2. **MEAN**：替換成該張影像的平均 RGB。

窗口邊界使用很小的 feather，寬度取 `min(4 px, 窗口短邊的 10%)`，避免硬邊界本身成為強特徵。

兩種遮罩都要跑。只有兩者得到相同趨勢，才把差異解讀成 context 效果；若結果相反，視為遮罩產生的 distribution shift，不能下結論。

### 4.3 Secondary intervention：crop-resize

將 R1/R2/R4/R8 區域 crop 後 resize 到 640×640 可作補充實驗，但不能與遮罩結果混在一起：

- 遮罩實驗：球的像素大小不變，主要測 context sufficiency。
- crop-resize：球被放大，同時改變 context 與解析度。

因此 crop-resize 若改善，只能說「局部放大有幫助」，不能單獨說大 receptive field 不重要。

### 4.4 如何建立可計算 AP(r) 的資料集

對每張影像，將該圖所有 ball 的 R1/R2/R4/R8 窗口取 union，union 外做遮罩：

- 保留原有所有 ball labels。
- 評估只使用 class `ball`。
- bat label 不列入該實驗的 AP/Recall，避免已被遮掉的 bat 被錯算成 false negative。
- 模型仍可輸出 ball false positives，照常計入 precision/AP。

如此每個 radius 都能產生與原 valid 一一對應的 intervention dataset，正式計算：

```text
AP50(r)
AP50-95(r)
Recall@IoU0.5(r)
Precision(r)
```

這是利用 ground truth 建立的診斷實驗，不是可部署的推論流程；報告中必須標示為 oracle-centered context ablation。

### 4.5 Object-level paired analysis

除了資料集級 AP，逐一配對同一顆 ball 在 FULL 與各 radius 的結果：

- class 必須為 `ball`。
- prediction 與 target IoU `>=0.5` 才算 detected。
- 記錄最佳匹配 confidence；沒有匹配時記為 0。
- 模型以低門檻 `conf=0.001` 推論，之後離線套用相同 confidence threshold。

主要 paired metrics：

1. **TP retention(r)**：FULL 中的 true positives，在半徑 r 仍被偵測的比例。
2. **Score retention(r)**：`score(r) / score(FULL)`。
3. **Lost TP(r)**：FULL 有偵測、r 條件下消失的數量。
4. **Rescued target(r)**：FULL 沒有、r 條件反而出現的數量；這通常代表遮罩效應，也必須揭露。

## 5. 使用哪個 validation set

原資料包含影片連續影格，初步檔名分組發現：

- train/valid 有 10 個相同原始 frame stem。
- train/valid 有 55 個來源群組重疊。
- 排除與 train 重疊的來源後，clean-valid 約 240 張、173 個 ball、189 個 bat。

因此：

1. 原 valid 567 張只用來與既有結果對照。
2. clean-valid 作主要結論。
3. bootstrap 必須以 source video/group 為抽樣單位，不能把相鄰 frames 當成獨立樣本。
4. 執行前人工抽查 source grouping，特別是 `youtube`、`Untitled_mov`、`*-MPEG-*`。

## 6. 場景與球狀態分層

平均 AP(r) 只能回答整體趨勢。為回答「哪些場景或球狀態不需要大 context」，至少分層：

- ball 尺寸：最短邊 `<8`、`8–16`、`16–32`、`>=32 px`。
- ball 與最近 bat bbox 的距離：相交／近／遠／無 bat。
- ball 局部 blur：以固定 patch 的 Laplacian variance 分位數分組。
- ball 到畫面邊界的距離。
- source group／拍攝視角。

每個 subgroup 畫 Recall(r) 與 TP retention(r) curve。少於 30 顆球的 subgroup 只作描述，不作確定結論。

可回答的例子：

- 清晰且孤立的球在 R2 已飽和。
- 模糊球需要 R4 才能利用球棒／手部資訊。
- 遠離 bat 的球在 R8 仍無改善，問題可能是解析度而非 context。

## 7. CPU 執行流程

### E0：建立 audit 與 target manifest

輸出至 `context_rf_cpu/`：

- `data_audit.json`
- `clean_valid.txt`
- `targets.csv`
- bbox/cell histograms
- source overlap 清單

不複製或修改 `../bbt5-detect-baseline/dataset`。

### E1：FULL baseline

使用既有 checkpoint：

`../bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`

在 CPU 跑 FULL inference，保存低 confidence threshold 的 predictions。此 checkpoint 是從已訓練 pose model 轉入相容 detection 權重，不是獨立重新訓練的 detect baseline，報告必須保留此限制。

驗收：

- 隨機視覺檢查至少 20 個 TP、FN、FP。
- 確認 ball per-class 指標，不使用 ball/bat 平均值代替。
- 固定 IoU、NMS、confidence 設定供所有 radius 共用。

### E2：R1/R2/R4/R8 遮罩 sweep

執行順序：

1. `GRAY × {R1,R2,R4,R8}`
2. `MEAN × {R1,R2,R4,R8}`
3. `FULL`

每個條件分 batch 推論並立即保存 prediction JSON。intervention image 預設在記憶體生成，只保存少量示意圖，避免產生大量重複影像。

### E3：分析與 bootstrap

輸出：

- 全體 ball 的 AP(r)、Recall(r)、Precision(r)。
- paired TP/score retention curve。
- 尺寸、blur、bat 距離、視角的 subgroup curves。
- 以 source group bootstrap 的 95% CI。

### E4：crop-resize sensitivity（可選）

只有遮罩結果已穩定後才跑。結果獨立成表，不納入 receptive-field 的主要判定。

## 8. 事先定義判定規則

相鄰 context 的改善需同時符合：

- Recall 或 TP retention 增益至少 3 percentage points。
- source-group bootstrap 95% CI 不支持只是微小／不穩定差異。
- GRAY 與 MEAN 趨勢一致。

### A. 大 receptive field 沒有必要性證據

若：

```text
R4 與 R8/FULL 的 Recall 差距 < 3 points
且 R4 的 TP retention >= 95%
```

則結論為：

> 超過 4× bbox context 後效能已飽和，額外大 receptive field 的收益有限。

### B. 需要較大局部 context

若 R8 相對 R4 穩定提升至少 3 points，但 FULL 相對 R8 沒有提升：

> 模型需要約 8× bbox 的局部場景，但沒有證據需要完整球場 context。

### C. 需要遠距／全圖 context

只有 FULL 相對 R8 穩定提升至少 5 points、兩種遮罩一致、且結果不由單一 source group 主導時，才判定：

> 遠距場景資訊對 ball 偵測有實質貢獻，縮小 receptive field 具有風險。

### D. 無法判定

若 GRAY/MEAN 結果相反、R1 到 R8 全部異常崩潰、或 FULL baseline TP 太少，視為遮罩 OOD 或 checkpoint 不足，不對 receptive field 下結論。

## 9. 與 layer／branch／channel 的關係

這兩個實驗先回答「資料與場景是否需要大 context」：

- cell 統計回答哪些 feature scale 可能太粗。
- Context Radius Experiment 回答哪些場景／球狀態需要多少 context。

它們不能單獨回答哪個 layer、branch、channel 可以移除。若結果顯示 R4 已飽和，才值得進入下一階段：

1. 對候選大-kernel branch 做 inference-time zero ablation。
2. 量測輸出變化與 ball recall，而不是只看 branch weight。
3. 最終刪除後仍須 fine-tune／retrain 驗證。

因本輪不想重新驗證 MAFM，工作在 Context Radius Experiment 得出結論後停止，不進入上述模型內部 ablation。

## 10. 最終輸出格式

主表：

| Radius | Mask | Ball AP50 | Ball AP50-95 | Recall@0.5 | TP retention | Median score retention | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R1 | GRAY |  |  |  |  |  |  |
| R2 | GRAY |  |  |  |  |  |  |
| R4 | GRAY |  |  |  |  |  |  |
| R8 | GRAY |  |  |  |  |  |  |
| R1 | MEAN |  |  |  |  |  |  |
| R2 | MEAN |  |  |  |  |  |  |
| R4 | MEAN |  |  |  |  |  |  |
| R8 | MEAN |  |  |  |  |  |  |
| FULL | NONE |  |  |  | 1.000 | 1.000 |  |

必要圖表：

- bbox width/height/area histograms。
- P2/P3/P4 cell-size histograms。
- AP(r)、Recall(r)、TP retention(r) curves。
- 各球尺寸與球狀態的 Recall(r) curves。
- R2/R4/R8/FULL 的 retained/lost detection 對照圖。

最後只回答 context 在哪一個 bbox 倍率飽和，以及哪些球狀態例外；不以單張 heatmap 或 MASF 論文結果代替量測。
