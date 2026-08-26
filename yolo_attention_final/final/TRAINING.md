# 演算法與微調方式

## Attention 演算法

完整公式、Bit-True 位寬與資料流另見
[BINARY_QK_PWL_ALGORITHM.md](BINARY_QK_PWL_ALGORITHM.md)。

交付模型只替換 YOLO26m 的兩個指定 Attention sites，而且兩處必須同時存在：

- `model.10.m.0.attn`
- `model.22.m.0.1.attn`

Q 與 K 先轉成 deterministic binary signs，在 identity 與 Walsh-Hadamard bases 中計算，再以固定 power-of-two coefficients 組合 score，並加上 decomposed 2D relative-position bias。

Normalization 先將每一 row 置中，再把 score rounding 與 saturation 成 `[-10, 0]` 的 signed Q8.8。接著用 20 個寬度 `0.5` 的 segments 定位，並在 21 個固定 UQ1.15 endpoints 之間線性插值。Numerator 是正式 Bit-True 路徑；denominator 仍採 exact floating-point row sum 與 reciprocal。

`yolo_attention/` 內的程式分工如下：

- `attention.py`：完整 Attention dataflow。
- `binary_basis.py`：Binary Q/K 與 Hadamard scoring。
- `normalization.py`：Float-PWL 與 Bit-True PWL normalization。
- `projection.py`：Q/K/V projection 與 channel layout。
- `relative_bias.py`：decomposed relative-position bias。

## 為什麼訓練用 Float-PWL

Bit-True forward 包含 Q8.8 rounding、saturation、integer cast 與離散 segment indexing，直接反向傳播無法對 bias 與 Q/K 產生有效的一般梯度。本專案沒有在 Bit-True reference 偷加 STE，而是把訓練與正式驗證分開：

- 訓練：固定使用可微分 Float-PWL surrogate，沿用相同的 `[-10, 0]`、20 segments 與固定 endpoints。
- Gate／選擇／交付：把 EMA `best.pt` 重新 materialize 成 Bit-True PWL，完整驗證 COCO2017 val 5,000 images。

因此 Float-PWL 指標只用來訓練與診斷，不能取代正式 Bit-True 結果。

## 實際微調流程

### 第一階段：Block LR sweep

從同一個保留 parent 建立三條獨立路徑：

| 路徑 | Attention LR | 相鄰 block LR | 最大 epochs | patience | 實跑 epochs |
|---|---:|---:|---:|---:|---:|
| x1 | `5e-6` | `1e-6` | 8 | 3 | 5 |
| x2 | `1e-5` | `2e-6` | 8 | 3 | 7 |
| x4 | `2e-5` | `4e-6` | 8 | 3 | 4 |

這一階段用來檢查原本 LR 是否過度保守。結果不是較高 LR 較好；x1 的實際 Bit-True mAP50-95 最高，所以 recovery 回到較低、較穩定的 discriminative LR。

### 第二階段：Neck/Detect recovery

從 gate 接受的 parent 往外擴大 trainable scope：Attention `5e-6`、相鄰 block `1e-6`、Neck/Detect `5e-7`。最大 16 epochs、patience 5，實際 6 epochs early stop。

### 第三階段：Backbone 最後 stage

在前一階段之上加入 Backbone 最後 stage，LR 只有 `1e-7`；其餘層維持 Attention `5e-6`、相鄰 block `1e-6`、Neck/Detect `5e-7`。最大 16 epochs、patience 5，實際 7 epochs early stop。

### 第四階段：Full-model recovery

最後允許全模型以同一組分層 LR 做 recovery，最遠的 Backbone 維持 `1e-7`。最大 20 epochs、patience 6，實際 8 epochs early stop。這是風險最高的測試，因此仍受 BatchNorm buffer lock 與 Bit-True rollback gate 約束。

## 所有階段共同設定

- Dataset：COCO2017 train 118,287 images；val 5,000 images。
- Optimizer：AdamW；batch 16；imgsz 640；workers 8；AMP。
- Reproducibility：deterministic；seed 0。
- Scheduler：constant LR；weight decay `5e-4`。
- Warmup：`warmup_epochs=0`、`warmup_bias_lr=0`，完全停用。
- BatchNorm：允許 scope 內 affine parameters 訓練，但 running mean、variance、counter 全部鎖定。
- Selection：只依實際 Bit-True checkpoint 的 mAP50-95。
- Rollback：child 相對直接 parent 下降超過 `0.001` 就拒絕，不把退化權重傳到下一階段。
- Safety：loss、checkpoint tensor 或正式 metric 出現 NaN/Inf 時 fail closed。

## 如何執行

在 `final/` 目錄先設定資料與裝置：

```bash
python run.py configure \
  --data-root /path/to/coco2017 \
  --device 0 \
  --batch 16 \
  --workers 8
```

接著檢查、預覽並執行：

```bash
python run.py doctor
python run.py check
python run.py recipe
python run.py train
python run.py train --execute
python run.py status
python run.py predict path/to/images --device 0 --save
```

訓練 wrapper 預設只 dry-run，必須明確加上 `--execute` 才會呼叫本交付包內的
production engine。重跑所需的歷史 Phase-B parent 已保存在
`artifacts/runs/s0-phase-b-bittrue/`，不需要上一層 repository。
