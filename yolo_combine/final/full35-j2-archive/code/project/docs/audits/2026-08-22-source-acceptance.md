# Full35 YOLO26 來源與 BBT5 資料驗收

- 日期：2026-08-22
- 來源 bundle：`/home/uxin/yolo/yolo_achitechure/achitechure_1/final/`
- Pose source：`/home/uxin/yolo/original/pose/dataset/`
- Detect view：`/home/uxin/yolo/original/pose/detect_dataset/`
- 決策：接受為融合工程權威來源；Full35-A2 為主線，Partial75-A2 為備案

## 結論

來源 bundle 的內容雜湊、鎖定環境、17 個 checkpoint、Detect forward、Pose26 graph graft、shared trunk tensor mapping 與 Pose backward 均通過。它可以作為 F0/F0.5/F1 的正式來源，但不是一個已完成的 Detect–Pose 融合模型；融合 module、雙資料 loader、task-aware loss routing 與 checkpoint contract 仍由 `yolo_combine` 實作。

## 融合工程驗證結果

- F0.5 與 F1 已實作；預設 architecture 為 Full35。
- F1 只保留一份 layers 0–22，參數由 45,580,762 降至 26,529,701。
- Full35 Detect 與 Pose task graphs 的 shared outputs 與獨立 graphs 逐 tensor 完全相等。
- 1:1 與 2:1 真實 COCO/BBT5 joint steps 均完成；2:1 Detect loss 先平均。
- P0 checkpoint 的 Pose26 head 嚴格移植 411/411 tensors，並完成 joint step。
- inherited attention/PWL 與 Full35 MASF 在 optimizer step 後無 drift。
- combined checkpoint 可安全 save/rebuild/load，且 F1 可 materialize 成官方 task graphs。
- 本專案完整測試：29 passed。

以上是工程正確性證據；尚未取代完整資料、640 輸入的正式 P0/F1 精度實驗。

## 通過項目

### Bundle 與環境

- `scripts/verify_bundle.py`：121/121 manifest entries PASS。
- Python：3.12。
- PyTorch：`2.11.0+cu128`。
- Ultralytics：`8.4.90`。
- CUDA：12.8，可見一張 NVIDIA GeForce RTX 5090（32,607 MiB）。
- `pip check`：無 broken requirements。
- `scripts/inspect_models.py`：9/9 Bit-True 與 8/8 Float checkpoint PASS。

### Full35-A2 graph

- Detect head：`nc=80`、`end2end=True`。
- Detect inputs：layers `(16,19,22)`，strides `(8,16,32)`。
- attention sites：`model.10.m.0.attn`、`model.22.m.0.1.attn`。
- Full35 MASF：graft 於 layer 16 的 P3 C3k2 output。
- Float/Bit-True Detect GPU forward：均輸出 `(1,300,6)`。

### Pose26 相容性

- YOLO26m `Pose26(nc=2, kpt_shape=(2,3))` 可套用相同 attention conversion 與 Full35 MASF graft。
- Detect source layers 0–22 到 Pose target：587/587 tensors compatible，0 missing、0 shape mismatch。
- Pose GPU forward：輸出 `(1,300,12)`。
- 使用一筆真實 BBT5 valid 標註計算 E2E/RLE loss 並 backward：PASS。
- 有限且非空梯度到達 trunk、attention、MASF `alpha` 與 Pose keypoint head。

## BBT5 lineage 核對

| Split | Images | Labels | Instances | Empty | Ball | Bat |
|---|---:|---:|---:|---:|---:|---:|
| train | 6,080 | 6,080 | 8,281 | 36 | 3,312 | 4,969 |
| valid | 567 | 567 | 785 | 2 | 301 | 484 |

- Pose labels 全部為 11 欄。
- two-class Detect labels全部為 5 欄，且逐行等於 Pose label 前 5 欄。
- COCO80 Detect view精確映射 `ball 0→32`、`bat 1→34`。
- 6,647 個檔案逐筆比較：0 derivation mismatch、0 broken image symlink。

## 已知但不阻塞實作的問題

1. Bundle 文件與 YAML 使用舊路徑 `/home/uxin0/...`；本機 root 是 `/home/uxin/...`。來源保持唯讀，由本專案 adapter 修正。
2. 原始 Roboflow split 有 10 個 source groups 跨 train/valid；正式精度不得直接宣稱無洩漏。
3. Pose labels 有 4 個極小負 keypoint coordinates；正式衍生 view 需 clamp 並留下 manifest。
4. BBT5 valid 有 COCO train source overlap；只能作歷史/相對比較。正式 finalist 需 leakage-safe split。
5. Bundle 記錄 `pytest: 48 passed`，但未包含 test files；本專案將建立自己的 interface-level tests。
6. Bundle 只有 Detect checkpoints，沒有 YOLO26 Pose26 baseline。P0 必須先建立與訓練；舊 YOLO11m Pose 只作歷史 baseline/teacher。
7. Bundle 沒有安裝 metadata；pickle module resolution 由 SourceBundle adapter 集中處理。

## 主線選擇

- 主線：Full35-A2。
- 備案：Partial75-A2。
- 正式訓練使用 Float-PWL checkpoint；Bit-True 用於對照與後續 materialization/驗證。

Partial75 只在 Full35 無法通過逐項 `baseline - 0.08` gate，或後續效率限制要求時
啟動；不與 Full35 同時作主線，也不把兩者權重混合。
