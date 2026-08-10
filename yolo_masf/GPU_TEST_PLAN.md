# MASF-YOLO 單一 GPU 測試與訓練清單

更新日期：2026-08-09

## 一、啟動原則

本文件是正式使用 GPU 前的審閱清單，目前不代表已啟動訓練。所有 MASF 工作只允許由單一 `masf-yolo-phase1.service` 依線性 DAG 執行，不可同時手動啟動個別模型。

開始前必須依序通過：

1. `yolo-p2-study.service` 已不是 active、activating 或 reloading。
2. 沒有其他 MASF service 或 `pipeline.lock` owner。
3. GPU 0 可用記憶體至少 24 GiB、利用率不超過 10%，連續三次、每次間隔 60 秒均符合。
4. CPU 全套測試、資料 audit、環境與權重雜湊全部通過。
5. 使用者已確認本文件的模型、資料、epochs、指標與通過標準。

任何一項未通過，pipeline 都保持等待或失敗停止，不可繞過 gate。

## 二、使用資料

### 正式架構比較資料

- 原始來源：`bbt5-detect-baseline/dataset/`
- 固定輸入：`artifacts/static-phase1/dataset/data.yaml`
- 類別：`0=ball`、`1=bat`
- 影像尺寸：訓練與評估均為 640 letterbox
- 分組數：1,128
- train：5,093 張
- validation：781 張
- test：773 張
- dataset SHA-256：`6e16c975941dbae2af174a2eb4b5424bffd4736c74aad56d424805da019b8fbc`
- 切分規則：以 video/session/camera/原圖 stem/影像內容雜湊分組後做固定 seed 42 的 80/10/10；train、val、test 的 group 與影像 hash 不可重疊。

validation 用於調參、比較與 M2/M3 選模；test 只能在 `selection.json` 凍結後執行一次，不得依 test 結果回頭選模型。

### 官方原始 validation 對照

`bbt5-detect-baseline/data.yaml` 的原始 valid split（567 張、785 instances）只用來重現資料夾權重及原 pose 權重的官方 Ultralytics 指標，不與上述 leakage-safe validation 混成同一張公平比較表。

### 初始化權重

- 路徑：`bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`
- SHA-256：`9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d`
- 類型：由 `../original/pose/weight/yolo11m_bat.pt` 轉成的 YOLO11m Detect 權重。
- 重要限制：此權重已看過 BBT5 資料，因此所有 B0、B1、MFAM、P3M 與 SP2 結果都標記 `data_exposed=true`；只能解讀為同源初始化下的架構消融，不能宣稱完全無資料暴露的泛化能力。

## 三、要測的模型

| ID | 結構 | 初始化 | 正式 epochs |
|---|---|---|---:|
| B0 | 原三尺度 P3/P4/P5 Detect，不訓練 | 資料夾 detect 權重 | 0 |
| B1 | 四尺度 P2/P3/P4/P5 Detect baseline | detect 權重；10 epoch freeze + 90 epoch full | 100 |
| M7 | P2：DW3、DW5、DW(1×7→7×1) | B1 canonical | 100 |
| M0 | P2：DW3、DW5、DW(1×7→7×1)、DW(1×9→9×1) | B1 canonical | 100 |
| M1 | P2：DW3、DW5 | B1 canonical | 100 |
| M2 | 1/2 channels 通過 M1，其餘 bypass | B1 canonical | 100 |
| M3 | 1/4 channels 通過 M1，其餘 bypass | B1 canonical | 100 |
| P3M | P2 Identity；P3：DW3、DW5、DW(1×7→7×1)，不含 9×9 | B1 canonical | 100 |
| SP2 | hidden 32 單類別輕量 P2 Ball head；P3/P4/P5 主 head 預測 Ball/Bat；aux loss 1.0 | B1 canonical | 100 |

M2 與 M3 是唯一 selection candidates；其他模型不得進入 `BEST_PARTIAL` 選擇。

## 四、怎麼測

### A. CPU 靜態與正確性測試

在不暴露 CUDA 的 subprocess 執行：

```bash
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m compileall -q masf_yolo tests
```

檢查內容：模型 registry、P2/P3 slot、每個 kernel shape、forward shape、stride、梯度、權重 semantic transfer、canonical save/strict reload、資料 leakage audit、COCO export、選模隔離、DAG 順序、單一 service 與 predecessor gate。

通過標準：零 failed、零 error；資料與來源權重 hash 必須完全相符。

SP2 另測：P2 只接收 Ball target；Bat 不可被分配到 P2 loss；P2 單類別輸出正確映射回共同兩類別空間；與 P3–P5 候選合併後的座標、confidence、class ID 與共同 NMS 正確；不得先計算完整四尺度兩類別 head 再乘 mask。

### B. CUDA acceptance gate

對 B1、M7、M0、M1、M2、M3、P3M、SP2 逐一執行：

1. 640×640 forward。
2. 使用一個含 Ball target 的 synthetic batch 計算 finite box/class/DFL loss。
3. AMP backward 與 SGD optimizer step。
4. 檢查所有應訓練參數 gradient finite。
5. 儲存 canonical CPU-float32 state dict，在全新 process strict reload。
6. 記錄 Params、peak allocated GPU memory、transfer matched/missing/mismatch 與 checkpoint hash。

通過標準：無 OOM、NaN、Inf、shape error；strict reload 成功；模型 stride 與 head 數符合契約。SP2 還必須同時放入 Ball 與 Bat 真實 target，證明 target routing 正確。

### C. 共同比較 batch probe

依 `16 → 8 → 4 → 2 → 1` 測試所有正式訓練模型，選出每個模型都能完成 AMP optimizer step 的最大共同 batch。所有正式模型使用同一 batch，不允許個別調整以取得不公平速度或精度。

通過標準：選定 batch 對所有模型均成功，結果保存於 `common_batch.json`。

### D. Smoke training

M7、M0、M1、M2、M3、P3M、SP2 各跑 3 epochs，資料使用正式 train/validation，但輸出放在 `smoke_runs/`，不得寫入正式論文比較。

檢查：資料讀取、loss 有限、epoch 正常完成、best/last checkpoint 存在、resume 可用、fresh-process strict reload 與一次 validation 成功。

通過標準：3 epochs 全部完成且無永久錯誤。若任一 smoke 失敗，單一 pipeline 停止，不繼續後面的正式訓練。

### E. 正式訓練

固定設定：YOLO11m、imgsz 640、SGD、momentum 0.937、cosine LR、deterministic、AMP、nbs 64、seed 42。

單一 pipeline 順序：

1. B1-A：freeze backbone 0–10，10 epochs，`lr0=0.01`。
2. B1-B：全模型解凍，90 epochs，`lr0=0.001`。
3. M7 acceptance、3-epoch smoke、100-epoch formal。
4. M0、M1、M2、M3、P3M、SP2 依序完成 smoke。
5. M0、M1、M2、M3、P3M、SP2 依序完成 100-epoch formal。
6. 每個正式 best EMA 轉 canonical checkpoint並 fresh-process strict reload。

正式 epochs 合計 800；smoke 合計 21；總計 821 epochs。若 SP2 正式 epochs 經使用者調整，總數同步重算。

### F. Validation、selection 與 test

對 B0、B1、M7、M0、M1、M2、M3、P3M、SP2 保存：mAP50-95、mAP50、mAP75、Precision、Recall、per-class AP、COCO AP_S/M/L、Ball AP/Recall、Ball AP_S/M/L、tiny-ball、8–16 px、>16 px、長寬比大於 2 的 blur proxy、各 bin GT 數量，以及 FP/FN 案例。

只以 validation 的 M2/M3 選出 `BEST_PARTIAL`。規則：兩者 mAP50-95 差不超過 0.002 且 Ball Recall 差小於 0.01 時優先 M3，否則使用既定排序。凍結 `selection.json` 後才執行 test。

### G. 硬體測試

全部模型測 Params、MACs/GFLOPs、peak activation、P2 activation、depthwise/1×1 operator 數、estimated feature traffic、GPU peak memory，以及固定 warmup/iterations 的 batch=1 latency。環境、precision、imgsz、同步方式與 power mode 必須相同。

SP2 必須額外與 B1 比較 P2 head 的實際 MACs、activation 與 latency，證明省算力來自未計算 Bat P2 tower，而非輸出後遮罩；另檢查 ONNX export/operator list 是否只含硬體友善的固定形狀運算。

## 五、已確認的固定決策

- SP2 hidden channels：32。
- SP2 auxiliary Ball loss：1.0。
- M0 保留 P2 的 9 分支；P3M 明確不含 9×9。
- B1 合計 100 epochs，其餘每個正式變體 100 epochs；smoke 各 3 epochs。
- latency：FP16、batch=1、warmup 100、同步量測 1,000 iterations，保存 mean、P50、P95。
- SP2 CPU target routing、Bat-only batch、finite loss/backward、decode、官方 NMS、B1 transfer、canonical/native reload 已通過。
- CPU 全套測試：140 passed、0 failed；compileall 通過。
- 640 靜態 profiling：B1 87.431 GFLOPs，SP2 80.528 GFLOPs；SP2 總 MACs 下降 7.895%。正式 GPU latency仍依 FP16 100/1,000 規格量測。

CPU 全套測試通過後，才建立並啟動唯一的 `masf-yolo-phase1.service`；服務仍會等待原 `yolo-p2-study.service` 結束。
