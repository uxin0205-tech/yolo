# YOLO26m baseline 與官方 COCO 指標核對

日期：2026-08-13

## 結論先行

本 repository 的正式模型確實是 **Ultralytics YOLO26m detection**，不是 YOLO11m 或 YOLO26n。本機 checkpoint `weights/yolo26m.pt` 內嵌的模型設定為 `scale: m`、`yaml_file: yolo26m.yaml`、`end2end: true`，而且 checkpoint 自帶的 `metrics/mAP50-95(B)` 是 `0.52456`，與官方公布的 YOLO26m E2E 指標 `52.5`（四捨五入後）一致。

但「官方 mAP」不是單一數字。YOLO26 有兩條 detection output path：

| 指標名稱 | 官方 YOLO26m COCO val2017 mAP50–95 | 意義 |
|---|---:|---|
| 官方 standard / Non-E2E | `0.531` | one-to-many head，加 NMS；官方表格的 `mAP val 50–95` |
| 官方 E2E | `0.525` | 預設 one-to-one head，不使用 NMS；官方表格的 `mAP val 50–95 (e2e)` |
| 本 repo `B26-FP` | `0.5179978354` | 同一個 YOLO26m checkpoint、640、COCO2017 val images，但由目前 repository evaluator 算出的 **Ultralytics internal metric** |

官方論文 Table 7 明確說明 `53.1` 是 one-to-many + NMS，而 `52.5` 是預設 one-to-one E2E；這兩者本來就相差約 `0.6` AP。[Ultralytics YOLO26 paper, Sec. 4.4 / Table 7](https://arxiv.org/html/2606.03748v1#S4.SS4)

因此，本研究應把數字命名為：

- `Official-NMS reference = 0.531`：外部官方參考值，不是目前實驗的直接 parent baseline。
- `Official-E2E reference = 0.525`：與目前 checkpoint 預設 E2E path 對應的官方參考值。
- `B26-FP local baseline = 0.517998`：所有 I/H/T5、recovery 與 normalization 實驗的主要公平比較基準，因為它們共用相同本地 dataset YAML、Ultralytics version 與 evaluator。

目前不能把 `B26-FP = 0.5180` 寫成「官方 YOLO26m mAP」，也不能直接以 `0.531 - 0.5180` 宣稱 binary attention 已掉 `1.3` AP；`0.531` 是另一條 Non-E2E + NMS path。即使和官方 E2E `0.525` 比較，仍需先解決下述 evaluator protocol 差異。

## 1. 官方 YOLO26m 報告的是什麼

Ultralytics 官方模型表列出 YOLO26m：輸入尺寸 640、standard mAP50–95 `53.1`、E2E mAP50–95 `52.5`、20.4M fused parameters 與 68.2B fused FLOPs。[Ultralytics YOLO26 官方文件](https://docs.ultralytics.com/models/yolo26/#performance-metrics)；[Ultralytics repository README](https://github.com/ultralytics/ultralytics#models)

官方 README 進一步限定這些 accuracy 數字是 **COCO val2017、single-model、single-scale**，並提供 `yolo val detect data=coco.yaml device=0` 作為重現入口。[Ultralytics repository README, Detection (COCO)](https://github.com/ultralytics/ultralytics#models)

官方論文對兩欄的定義比 README 更完整：

- standard mAP：`end2end=False`，使用 one-to-many branch 和 NMS；
- E2E mAP：使用預設 one-to-one branch，真正 NMS-free；
- YOLO26m 在兩條路徑分別為 `53.1` 與 `52.5`。

來源：[Ultralytics YOLO26 paper, Table 7 and Sec. 4.4](https://arxiv.org/html/2606.03748v1#S4.SS4)。

這也解釋了為什麼「官方 mAP」看起來有兩個數字：它不是 COCO metric 有兩種定義，而是同一模型的兩條 inference head/path 都用 COCO AP 評估。

官方 COCO dataset 設定中的 validation split 是 `val2017.txt`，共 5,000 張影像；train2017 則為 118,287 張。[Ultralytics COCO dataset documentation](https://docs.ultralytics.com/datasets/detect/coco/#dataset-structure)；[官方 `coco.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

官方 Val API 將 `metrics.box.map` 定義為 box mAP50–95，`metrics.box.map50` 為 mAP50，`metrics.box.map75` 為 mAP75。[Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/#usage-examples)

## 2. 本 repo 的 `B26-FP` 到底是什麼

`B26-FP` 是 **未替換 attention 的原始 YOLO26m checkpoint，在本研究固定 evaluator 下重新驗證的 control run**。它不是重新訓練得到的新 baseline，也不是官方表格數字的人工抄錄。

本地量測 artifact 為 [`artifacts/runs/b26-fp/metrics/queue-result.json`](../../artifacts/runs/b26-fp/metrics/queue-result.json)：

| 本地量測 | 數值 |
|---|---:|
| mAP50–95 | `0.5179978354321163` |
| mAP50 | `0.6907129398442434` |
| mAP75 | `0.5666710540300949` |

本地 evaluation recipe 為 [`configs/evaluation/coco2017.yaml`](../../configs/evaluation/coco2017.yaml)：

- `data: data/coco2017.yaml`
- `imgsz: 640`
- `batch: 16`
- `device: 0`
- `workers: 8`
- `split: val`
- `plots: false`

本地 [`data/coco2017.yaml`](../../data/coco2017.yaml) 指向 `/home/uxin/yolo/coco2017`，其 validation entry 是資料夾 `images/val2017`。這確實是 COCO2017 val 的 5,000 張影像，但它和官方 `coco.yaml` 的 `val2017.txt` 表達方式不同。

checkpoint provenance（由本機 `weights/yolo26m.pt` 讀取）：

| 欄位 | 值 |
|---|---|
| file size | `44,255,705` bytes |
| SHA-256 | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` |
| model YAML | `yolo26m.yaml` |
| scale | `m` |
| classes | `80` |
| end-to-end | `true` |
| checkpoint Ultralytics version | `8.3.222` |
| checkpoint source commit | `557e95b99c24c5b0d4eb5973d5d54a20a799566a` |
| embedded mAP50–95 | `0.52456` |
| local evaluation Ultralytics version | `8.4.90` |

其中 embedded `0.52456` 可四捨五入成官方 E2E `52.5`，是「checkpoint 與官方 YOLO26m E2E 權重相符」的強證據；但 embedded training metric 不是本次 queue 重新量測值，因此仍須分欄保存，不能替代 `B26-FP` artifact。

## 3. 為什麼本地 `0.5180` 不等於官方 `0.525/0.531`

### 3.1 最重要差異：E2E 與 Non-E2E

本地 checkpoint 的 `end2end=true`，所以目前 `B26-FP` 對應 one-to-one E2E 路徑。官方 `0.531` 則是 one-to-many + NMS。合理的一級比較應先看：

$$
0.525 - 0.5179978 \approx 0.0070,
$$

也就是相對官方 E2E 約低 `0.70` AP，而不是拿 Non-E2E `0.531` 算出 `1.30` AP 後全部歸因於模型或 attention。

### 3.2 本地 evaluator 沒有進 canonical COCO JSON API path

這是目前最關鍵的 protocol 問題。Ultralytics `DetectionValidator` 只有在 validation path 同時：

1. 含有 `coco`；且
2. 結尾是 `val2017.txt` 或 `test-dev2017.txt`

時才設 `is_coco=True`。接著只有在 COCO/LVIS、`save_json` 生效且存在 predictions 時，才以 `instances_val2017.json` 呼叫 `faster-coco-eval`，並用 `AP_all` 覆寫 mAP50–95、用 `AP_50` 覆寫 mAP50。[Ultralytics official `detect/val.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py)

本 repo 使用的 entry 是 `images/val2017`，不是 `val2017.txt`，因此目前 Ultralytics 8.4.90 不會把它辨認成 canonical COCO config；`B26-FP = 0.517998` 是 Ultralytics internal detection metric，而不是由 COCO annotation JSON / `faster-coco-eval` 回填的 canonical COCO metric。

這不表示本地 baseline 無效。它仍是目前所有研究 variants 最公平的**內部控制基準**，因為所有方法使用同一組影像與同一 evaluator。但它不能單獨用來宣稱已重現官方 COCO benchmark。

### 3.3 其他會影響精確重現的變數

Ultralytics 官方 Val 文件明確允許調整 dataset、`imgsz`、batch、confidence、IoU 與 device，因此重現時應保存全部 validation arguments，而不能只記模型名稱。[Ultralytics validation arguments](https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation)

本地與官方至少還存在以下可能差異：

- checkpoint 記錄的 framework 是 Ultralytics 8.3.222，本地重測為 8.4.90；validator、post-processing 或 metric implementation 可能隨版本改動；
- 官方數字以官方 `coco.yaml` 與 canonical COCO JSON evaluator 為準，本地目前走 internal metric；
- standard 與 E2E 的 head、是否 NMS 必須明確指定；
- batch 通常不應改變理想精度，但仍應記錄以確保完整 provenance；device 的浮點數值差異通常很小，不足以在沒有證據時解釋全部 `0.70` AP。

最後一點是推論，而不是官方明示原因；目前證據足以定位 evaluator protocol 不一致，但不足以把 `0.70` AP 精確分解到每一項因素。

## 4. 本研究應如何使用 baseline

### 研究內部比較

主比較一律使用同一 evaluator 下的 `B26-FP = 0.517998`：

$$
\Delta\mathrm{mAP}_{method}
=
\mathrm{mAP}_{method,local}
-
\mathrm{mAP}_{B26\text{-}FP,local}.
$$

這樣 I/H/T5 或 Direct/Progressive 的差異才不會混入 official-vs-local protocol change。

### 對外報告

報告表格應同時保留：

| 欄位 | 應填內容 |
|---|---|
| Official YOLO26m NMS reference | `0.531` |
| Official YOLO26m E2E reference | `0.525` |
| Local parent baseline | `0.517998` |
| Local evaluator | Ultralytics internal metric，COCO2017 val images，640，E2E |
| Variant delta | 相對 local parent，而不是直接相對官方 NMS |

在補做 canonical COCO API 驗證以前，建議將目前欄位命名為 `local mAP50-95 (Ultralytics internal)`，而不是 `official COCO AP`。

## 5. 若要正式核對官方數字，最小必要重測

後續若獲准再執行 GPU validation，應用官方 `coco.yaml`／`val2017.txt` 讓 validator 進入 `is_coco=True`，並保存 predictions、完整 args、Ultralytics version 與 checkpoint hash。至少測兩條 path：

```text
V-OFFICIAL-E2E：yolo26m.pt，end2end=True，640，COCO val2017，COCO JSON evaluator
V-OFFICIAL-NMS：yolo26m.pt，end2end=False，640，COCO val2017，COCO JSON evaluator
```

預期對照值分別是 `0.525` 與 `0.531`；但在真正量測前應標示 `not_run`，不可把 checkpoint embedded metric 或官方表格複製成新的本地結果。

這組重測只負責 **benchmark protocol validation**。既有 `B26-FP` 仍應保留，不得覆寫，因為它是已完成 I/H/T5 與 recovery 實驗的共同 local parent。

## Sources

- [Ultralytics YOLO26 official model documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics YOLO26 paper, arXiv:2606.03748](https://arxiv.org/html/2606.03748v1)
- [Ultralytics official repository README](https://github.com/ultralytics/ultralytics#models)
- [Ultralytics COCO dataset documentation](https://docs.ultralytics.com/datasets/detect/coco/)
- [Ultralytics official `coco.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)
- [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/)
- [Ultralytics official detection validator source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py)
