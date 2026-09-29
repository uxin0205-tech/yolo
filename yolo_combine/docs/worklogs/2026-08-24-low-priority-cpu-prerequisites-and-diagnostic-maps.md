# 2026-08-24 低優先序 CPU 前置驗證與診斷 mAP

## 目的

在 RTX 5090 已有其他正式訓練工作的情況下，先完成不需要 GPU 的 Full35 前置查核、
canonical BBAT5 trainer smoke、可重用的獨立 Detect 評估，以及舊 Pose P1 的 Float／Bit-True
診斷。這一輪不解除正式 preflight 的兩個 blocker，也不把 CPU smoke 當成 P1→P2→P3。

## 資源隔離與原因

所有較長命令都使用以下限制：

```text
nice -n 19
ionice -c 3
taskset -c 22-23
CUDA_VISIBLE_DEVICES=-1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
workers=0
```

因此命令看不到 CUDA，只在兩個 CPU core 上以最低 CPU／I/O 排程優先權執行。執行後主機
仍有約 75–77 GiB available RAM；GPU compute process 仍只有既有 MambaPose PID 1720123，
約使用 21,786 MiB，沒有本專案 GPU process。全量 COCO Float validation 的最高 RSS 為
2,691,068 KiB（約 2.57 GiB）；Bit-True 為 2,948,108 KiB（約 2.81 GiB）。

## 完成內容與結果

### 1. workspace、資料與 shared model CPU check

- `audit --verify-hashes` 通過：Full35 source hash 無 missing 或 mismatch。
- canonical `bbat5-v1` 通過：train 5,964／val 683、source group overlap 0、broken link 0、
  derivation mismatch 0。
- `cpu-check --imgsz 64 --seed 0` 通過：Detect 2＋Pose 1 的 macro-step 有 642 個有限梯度
  tensor，沒有 NaN／Inf。
- 兩個獨立模型為 45,580,762 parameters；共享模型為 26,529,701，移除重複參數
  19,051,061（41.796%）。Float 與 Bit-True 的 Detect／Pose 輸出均與對應獨立模型逐值
  相同。
- 報告：`variants/full35/artifacts/cpu-validation.json`。

### 2. canonical BBAT5 原生 Pose trainer smoke

以 canonical runtime View、CPU、imgsz 128、batch 2、fraction 0.001、6 張 train image、
1 epoch 跑完原生 Pose trainer 與 checkpoint 儲存：

```text
variants/full35/artifacts/pose/full35-pose-canonical-cpu-smoke-20260824/
```

產生 `best.pt`、`last.pt`、`results.csv`。Ultralytics 即使設定 `val=false`，訓練結束時仍對
683 張 canonical val 執行 final evaluation；此行為已確認。由於這是 fresh head 加上只有
6 張訓練影像，數值只證明資料、loss、trainer、checkpoint 與 metric pipeline 可執行：

| 指標 | 結果 |
| --- | ---: |
| Box mAP50 | 0.00002 |
| Box mAP50-95 | 約 0.00000 |
| Pose mAP50 | 0.01890 |
| Pose mAP50-95 | 0.01168 |

這些數字不得作正式 baseline 或精度結論。

### 3. 舊 Pose P1 在 canonical val 的診斷

使用舊 checkpoint
`artifacts/runs/p0/p0-full35-p1-seed0/weights/best.pt`，在 canonical 683 張 val、
imgsz 640、batch 2、CPU 分別跑 Float 與 Bit-True。此 checkpoint 是 legacy basic split、
head-only P1，可能與 canonical val 有 train／val lineage overlap，因此兩份 JSON 都明確標為
`formal_gate_eligible=false`。

| canonical BBAT5 指標 | Float | Bit-True | Bit-True − Float |
| --- | ---: | ---: | ---: |
| Overall Box mAP50-95 | 0.5767147 | 0.5767323 | +0.0000176 |
| Overall Pose mAP50-95 | 0.8797845 | 0.8798075 | +0.0000229 |
| Ball Box mAP50-95 | 0.4978672 | 0.4979129 | +0.0000458 |
| Ball Pose mAP50-95 | 0.8666584 | 0.8666883 | +0.0000298 |
| Bat Box mAP50-95 | 0.6555623 | 0.6555517 | −0.0000106 |
| Bat Pose mAP50-95 | 0.8929106 | 0.8929266 | +0.0000161 |

這只表示舊 P1 在此診斷中的 Float／Bit-True representation drift 很小，不能取代 canonical
P3 baseline。完整 P／R、mAP50、mAP75、mAP50-95 位於：

- `variants/full35/artifacts/diagnostics/provisional-p1-canonical-cpu-float/metrics.json`
- `variants/full35/artifacts/diagnostics/provisional-p1-canonical-cpu-bittrue/metrics.json`

### 4. 獨立 Full35 Detect 的 COCO Float 前置評估

使用獨立 Float checkpoint
`/home/uxin/yolo/yolo_achitechure/achitechure_1/final/weights/float/full35-a2.pt`，在完整
COCO2017 val 5,000 張、imgsz 640、batch 2、CPU 完成評估。耗時約 39 分 28 秒，最高 RSS
約 2.57 GiB。

| Ultralytics internal 指標 | 結果 |
| --- | ---: |
| COCO80 Box mAP50 | 0.6762031 |
| COCO80 Box mAP50-95 | 0.5062334 |
| COCO80 Precision | 0.7298898 |
| COCO80 Recall | 0.6143798 |
| Person Box mAP50 | 0.8411002 |
| Person Box mAP50-95 | 0.6256977 |
| Person Precision | 0.7941614 |
| Person Recall | 0.7689856 |

同一次 Faster-COCO-Eval 的 canonical aggregate raw 值為 COCO80 mAP50 0.6840032、
mAP50-95 0.5122113；兩套 evaluator 口徑都原樣保存，不混成同一 gate。產物位於：

```text
variants/full35/artifacts/standalone-baseline-precompute/detect-cpu/float/
```

此 JSON 標為 `independent_detect_partial_baseline_candidate`；它可保留作 Detect Float 前置
資料，但不能單獨解除八項 baseline blocker。

### 5. 獨立 Full35 Detect 的 COCO Bit-True 全量評估

使用獨立 Bit-True checkpoint
`/home/uxin/yolo/yolo_achitechure/achitechure_1/final/weights/bittrue/full35-a2.pt`，在相同
5,000 張 COCO val、imgsz 640、batch 2、CPU 完成評估。耗時 39 分 17 秒、最高 RSS 約
2.81 GiB、無 swap；產生 586,325 筆 prediction。checkpoint SHA256 為
`5ca57319d3708e5639c4852bac0898038d61ac793a4973015f1b6be8b764582d`。

| Ultralytics internal 指標 | Float | Bit-True | Bit-True − Float |
| --- | ---: | ---: | ---: |
| COCO80 Box mAP50 | 0.6762031 | 0.6761958 | −0.0000073 |
| COCO80 Box mAP75 | 0.5544923 | 0.5545003 | +0.0000081 |
| COCO80 Box mAP50-95 | 0.5062334 | 0.5062554 | +0.0000219 |
| Person Box mAP50 | 0.8411002 | 0.8410926 | −0.0000076 |
| Person Box mAP75 | 0.6842974 | 0.6842537 | −0.0000436 |
| Person Box mAP50-95 | 0.6256977 | 0.6256870 | −0.0000107 |
| Person Precision | 0.7941614 | 0.7940900 | −0.0000714 |
| Person Recall | 0.7689856 | 0.7687668 | −0.0002188 |

Faster-COCO-Eval 1.7.2 的 canonical 結果：

| canonical COCO 指標 | Float | Bit-True | Bit-True − Float |
| --- | ---: | ---: | ---: |
| AP50-95 | 0.5122113 | 0.5122265 | +0.0000151 |
| AP50 | 0.6840032 | 0.6839963 | −0.0000070 |
| AP75 | 0.5601221 | 0.5601321 | +0.0000100 |
| AP small | 0.3518008 | 0.3518036 | +0.0000028 |
| AP medium | 0.5564513 | 0.5564463 | −0.0000049 |
| AP large | 0.6583425 | 0.6583454 | +0.0000030 |
| AR maxDets=100 | 0.7139216 | 0.7139167 | −0.0000050 |

所有 AP 差異遠小於 0.08 gate；這證明此獨立 Detect checkpoint 的 Float／Bit-True
representation drift 可忽略，但不是 JOINT 精度結論。產物位於：

- `variants/full35/artifacts/standalone-baseline-precompute/detect-cpu/bittrue/metrics.json`
- `variants/full35/artifacts/standalone-baseline-precompute/detect-cpu/bittrue/coco-eval.json`
- `variants/full35/artifacts/standalone-baseline-precompute/detect-cpu/representation-delta.json`

### 6. patience 唯讀核對

| run／stage | 最大 epoch | patience | 實際語意 |
| --- | ---: | ---: | --- |
| 歷史 Full35 Detect A2 | 10 | 4 | 實跑 5 epoch 後 early-stop |
| 歷史 legacy Pose P1 | 10 | 5 | 僅歷史 basic split；不沿用 |
| 正式 Pose P1 | 17 | 0 | 關閉 early stopping，跑滿 17 |
| 正式 Pose P2 | 22 | 0 | 關閉 early stopping，跑滿 22 |
| 正式 Pose P3 | 100 | 20 | 最佳指標連續 20 epoch 未改善才停止 |
| JOINT J1 | 5 | 0 | 關閉 early stopping，跑滿 5 |
| JOINT J2 | 40 | 0 | 關閉 early stopping，跑滿 40 |
| 選配 JOINT J3 | 20 | 5 | `best_joint` 連續 5 epoch 未改善才停止 |

J3 預設關閉。這次 COCO validation 沒有 optimizer 或訓練 epoch，因此不使用 patience。

## 困難與解法

1. 困難：完整 COCO CPU validation 每個 backend 約需 40 分鐘，可能與既有 GPU 訓練的
   CPU preprocessing／I/O 競爭。
   解法：先停在 Float；取得使用者明確授權後，才以相同 CUDA 隱藏、兩核心、`nice=19`
   與 idle I/O 完成 Bit-True。期間持續監看 GPU PID、RAM 與 load，沒有啟動本專案 GPU
   process。
2. 困難：現有 Pose checkpoint 只有 legacy basic-split P1，並非 canonical P3。
   解法：只做明確標記的 diagnostic，不寫入 formal gate 路徑、不修改 `joint.yaml`。
3. 困難：`val=false` 仍觸發 Ultralytics training-end final evaluation。
   解法：保留此結果並記錄行為；正式 smoke 時不能把 `val=false` 解讀為完全不驗證。
4. 困難：Ultralytics validator 回傳字典沒有持久化 canonical AP75 與完整 AR，但終端會顯示。
   解法：用相同 faster-coco-eval 1.7.2 從保存的 predictions 重算 Float／Bit-True，另存
   `coco-eval.json`，不重新推論也不混用 internal／canonical 口徑。

## 未解事項與風險

- canonical BBAT5 Full35 Pose P1 17 → P2 22 → P3 最多 100 尚未執行；仍須等 GPU 空閒。
- formal physical batch 128 尚須在空閒 RTX 5090 上做 memory profile，75 GiB system RAM
  不能證明 32 GiB VRAM 可容納。
- Detect Float／Bit-True 前置 AP 已完成；完整八項 baseline 仍缺 canonical Pose P3，並須由
  正式 baseline 流程把 Detect 與該 P3 結果綁成同一份 gate artifact。
- `preflight` 應繼續 fail-fast；本輪沒有繞過或放寬 gate。
- 正式訓練完成後會產出每 epoch 的 Box/Pose precision、recall、mAP50、mAP50-95；正式
  baseline 與 joint validation 另會保存 overall／per-class JSON、CSV 與圖表。

除上述已記錄事項外，無其他新增困難。
