# 已完成 Activation 整合報告

日期：2026-08-30
適用模型：YOLO26m Full35 shared trunk + COCO80 Detect head + BBAT5 Pose26 head

## 1. 這份報告回答什麼

這份報告只整理「本專案實際完成、實作或評估過的 activation」，不把計畫中的工作寫成已完成成果。
目前 registry 共有六種 activation：

1. `silu`：正式父模型與精度控制組。
2. `hardswish`：既有硬體鄰近 baseline。
3. `relu`：最低成本的診斷控制組。
4. `qsilu_pq`：本專案凍結的 C¹、dyadic 分段二次 SiLU 近似。
5. `poly_quality`：SIPA 的品質導向 C² 積分多項式。
6. `poly_shift`：SIPA 的 dyadic／APoT C² 積分多項式。

BCSP 不是第七種 activation；它是逐區域 activation placement 的 bounded search 支援架構。本輪完成
BCSP 的 manifest、region、queue 與 gate propagation，但沒有得到實際 mixed-policy 搜尋結果。

## 2. 最終狀態總表

| Activation | 實作 | Full zero-shot | 10-epoch recovery | Finalist | 可交付權重 | 目前定位 |
| --- | --- | --- | --- | --- | --- | --- |
| SiLU | PyTorch／原模型 | accepted baseline | sham control 已跑 | seed-1 control 完成，epoch 6 early-stop | 有 | 正式基準與量化 control |
| Hardswish | PyTorch standard op | 完成、失敗 | 完成、失敗 | 排除 | 未發布 | 硬體 baseline／負結果 |
| ReLU | PyTorch standard op | 完成、八項皆 0 | 未啟動 | 排除 | 未發布 | primitive cost floor |
| `qsilu_pq` | PyTorch／Q16.10／ONNX | 完成、通過 | 完成、通過 | 人工停止，只有 provisional epoch 1 | 有 10-epoch 權重 | 唯一保留的非 SiLU 量化候選 |
| `poly_quality` | PyTorch float reference | 完成、通過 | 完成、失敗 | 排除 | 未發布 | SIPA 品質 profile／負結果 |
| `poly_shift` | PyTorch／Q16.10／ONNX | 完成、失敗 | 完成、失敗 | 排除 | 未發布 | SIPA shift profile／硬體數學對照 |

因此最重要的結論是：

> 已完成的 non-SiLU 候選中，只有 `qsilu_pq` 通過完整 COCO2017 + Canonical BBAT5 v1 的
> 10-epoch 八項 accuracy gate；下一階段應比較 accepted SiLU 與 qSiLU 的量化行為。

## 3. 六種 Activation 的可讀公式

完整逐步推導放在[可讀版數學文件](../docs/research/full35-activation-mathematical-derivations.md)。本節只保留
閱讀實驗結果需要的公式與直觀意義。

### 3.1 共用骨架

令 `u = |x|`，也就是先取輸入的絕對值。主要候選都採用：

```text
輸出 A(x) = x / 2 + 修正量 H(|x|)
```

對 `x` 與 `-x`，修正量完全相同，所以相減時會抵消：

```text
A(x) - A(-x)
= [x/2 + H(|x|)] - [-x/2 + H(|x|)]
= x
```

如果 `H(0)=0`，輸入 0 時輸出也是 0。如果門檻 `T` 以外令 `H(u)=u/2`，正尾端輸出就是 `x`，
負尾端輸出就是 0，也就是精確 ReLU 尾端。

### 3.2 SiLU

```text
SiLU(x) = x × sigmoid(x)
```

也可以改寫成共用骨架：

```text
SiLU(x) = x/2 + [|x|/2] × tanh(|x|/2)
```

SiLU 是正式父模型使用的 activation。它光滑、保留負輸出谷與非單調區域；硬體若沒有 fused SiLU，
通常需要 sigmoid／exp 類運算。

### 3.3 Hardswish

| 輸入範圍 | 輸出 |
| --- | --- |
| `x ≤ -3` | `0` |
| `-3 < x < 3` | `x × (x + 3) / 6` |
| `x ≥ 3` | `x` |

中央區間是二次式，兩端精確等於 ReLU。它只有 C0，門檻上的一階導數不連續。雖然常見 backend 有
fused operator，本模型直接替換 190 個 sites 後仍產生很大的 feature drift。

### 3.4 ReLU

| 輸入範圍 | 輸出 |
| --- | --- |
| `x < 0` | `0` |
| `x ≥ 0` | `x` |

ReLU 是最低成本控制組，但拿掉 SiLU 的負谷與平滑轉換後，本次 zero-shot 八項 mAP50-95 全為 0。

### 3.5 qSiLU：`qsilu_pq`

qSiLU 仍使用 `qSiLU(x)=x/2+h(|x|)`。令 `u=|x|`，程式實際使用下列分段：

| `u` 的範圍 | 修正量 `h(u)` |
| --- | --- |
| `0 ≤ u < 1` | `(57/256)u²` |
| `1 ≤ u < 2` | `(23/256)u² + (17/64)u - 17/128` |
| `2 ≤ u < 4` | `-(11/512)u² + (91/128)u - 37/64` |
| `4 ≤ u < 8` | `-(5/1024)u² + (37/64)u - 5/16` |
| `u ≥ 8` | `u/2` |

- 門檻是 1、2、4、8，全部是 2 的冪。
- 全部係數都是 dyadic，可用 shift／add 表示常數。
- 四個區段可共用一次 `u²`。
- 全域 C1，保留負谷與非單調導數；`|x|≥8` 時精確等於 ReLU。
- `[-8,8]` 對 SiLU：MAE 0.0051297、RMSE 0.0060978、最大誤差 0.0108319。
- 已通過 float、Q16.10 bit-true、ONNX 介面與 Full35 10-epoch accuracy gate。
- 「硬體友善」目前是結構判斷，尚未有指定 FPGA／ASIC 的 latency、power 或資源實測。

### 3.6 SIPA 一般設計

SIPA 先把中央區間縮放成 `z=u/T`，再設計修正量的斜率 `q_a(z)`：

```text
q_a(z) = a z
       + (9 - 9a/2) z²
       + (6a - 16) z³
       + (15/2 - 5a/2) z⁴
```

四個限制不要連成一行看；它們分別代表：

1. `q_a(0)=0`：原點的修正斜率是 0。
2. `q_a(1)=1/2`：中央區間走到尾端時，斜率接到 `u/2` 直線。
3. `q_a` 在 `z=1` 的導數等於 0：曲率接到尾端的 0。
4. `q_a` 從 0 到 1 的曲線下面積等於 1/2：函數值走到尾端時剛好是 `T/2`。

累積斜率後得到：

```text
R_a(z) = (a/2) z²
       + (3 - 3a/2) z³
       + (3a/2 - 4) z⁴
       + (3/2 - a/2) z⁵
```

最後的修正量分成兩段：

| 輸入大小 | `H(u)` |
| --- | --- |
| `0 ≤ u ≤ T` | `T × R_a(u/T)` |
| `u ≥ T` | `u/2` |

這四個限制共同保證 zero anchor、symmetry invariant、C2 接合與精確 ReLU 尾端。

#### `poly_quality`

```text
a = 4
T = 109/16 = 6.8125

H(u) = (32/109)u²
     - (768/11881)u³
     + (8192/1295029)u⁴
     - (32768/141158161)u⁵       ，0 ≤ u ≤ 109/16

H(u) = u/2                      ，u ≥ 109/16
```

負谷約為 -0.27904，接近原始 SiLU。它的 zero-shot 很接近 baseline，但係數不是 dyadic，而且
10-epoch recovery 的 BBAT ball box 未通過 gate。

#### `poly_shift`

```text
a = 9/2
T = 8

H(u) = (9/32)u²
     - (15/256)u³
     + (11/2048)u⁴
     - (3/16384)u⁵       ，0 ≤ u ≤ 8

H(u) = u/2               ，u ≥ 8
```

直接係數全部是 dyadic，但仍需形成 `u²` 到 `u⁵`，不是零乘法器設計。它完成 11-region zero-shot
sensitivity，但 uniform 10-epoch gate 失敗，所以後續 BCSP jobs 依規則封鎖。

### 3.7 Fixed-point 不變量

令 `n` 是 fixed-point 的整數輸入 code，正負輸入共用同一個 `H_hat(|n|)`：

```text
A_hat(n) = floor(n/2) + H_hat(|n|)
```

因為 `floor(n/2)-floor(-n/2)=n`，而 `H_hat(|n|)` 會相減抵消，所以：

```text
A_hat(n) - A_hat(-n) = n
```

尾端若使用 `H_hat(u)=ceil(u/2)`，奇數 code 的正、負 ReLU 尾端也能保持 0 LSB 誤差。這是後續量化
工作要做全輸入碼枚舉的 bit-exact 性質，不代表 PTQ／QAT AP 已完成。

## 4. 模型、資料與比較契約

| 項目 | 固定值 |
| --- | --- |
| 父模型 | YOLO26m Full35 shared trunk + COCO80 Detect + BBAT5 Pose26 |
| 父權重 | `yolo_combine/final/full35/weights/combined/inference/best_joint.pt` |
| 父權重 SHA-256 | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| COCO2017 | 118,287 train／5,000 val |
| Canonical BBAT5 v1 | 5,964 formal train／683 formal val |
| 影像尺寸 | 640 |
| 資料比例 | `fraction=1.0`、`resampling=false` |
| Activation sites | 190 references、11 regions |
| Selector backend | Bit-True |
| Accuracy gate | 八項 mAP50-95 每一項 delta 都必須 ≥ -0.015 |

沒有建立 30% 資料夾、重新切分或修改 BBAT5 樣本／標註。COCO 與 BBAT5 raw AP 不互相平均；每項
指標都直接對 accepted SiLU baseline 計算 delta。

## 5. 訓練設定

| 階段 | Epoch | Seed | Patience／warm-up | LR scale | 狀態 |
| --- | ---: | ---: | --- | ---: | --- |
| Baseline／zero-shot／profiling | 0 | 0／1 | — | — | 完成 |
| Uniform short recovery | 10 | 1 | 0／1 | 原 J3 的 0.1× | 四候選完成 |
| Region sensitivity／BCSP search | 10 | 1 | 0／1 | 原 J3 的 0.1× | 14 jobs gate-blocked |
| Finalist seed 1 | 最多 20 | 1 | 5／3 | 原 J3 的 1.0× | SiLU 完成；qSiLU 人工停止 |
| PTQ | 0 | 1 | calibration | — | 尚未在本專案執行 |
| QAT if needed | 10 | 1 | 0／1 | 原 J3 的 0.5× | 未啟動 |

Recovery 共用 AdamW、weight decay 2.7×10⁻⁴、betas (0.948, 0.999)、AMP、gradient clip
10 與 cosine final factor 0.5。Short recovery LR：backbone 3.8e-7、neck 1.9e-6、MASF 3.8e-6、
attention 5e-8、Detect／Pose heads 5e-6。

Detect logical batch 固定 128。qSiLU physical batch 128、64、32 都在第一個 forward OOM；physical
16 可穩定完成，因此以梯度累積維持 logical batch。這是目前 eager PyTorch reference 在 31.35 GiB
GPU 的實測，不是 fused kernel 的理論上限。

## 6. Accepted baseline

| 指標 | mAP50-95 |
| --- | ---: |
| COCO box | 0.498022 |
| COCO person box | 0.620381 |
| BBAT box | 0.630036 |
| BBAT pose | 0.903717 |
| BBAT ball box | 0.507437 |
| BBAT bat box | 0.752634 |
| BBAT ball pose | 0.859909 |
| BBAT bat pose | 0.947526 |

## 7. Uniform zero-shot

| Activation | COCO Δ | Person Δ | BBAT box Δ | BBAT pose Δ | Ball box Δ | Bat box Δ | Ball pose Δ | Bat pose Δ | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Hardswish | -0.077698 | -0.068644 | -0.159332 | -0.027252 | -0.142435 | -0.176230 | -0.040182 | -0.014323 | 失敗 |
| ReLU | -0.498022 | -0.620381 | -0.630036 | -0.903717 | -0.507437 | -0.752634 | -0.859909 | -0.947526 | 失敗 |
| `qsilu_pq` | -0.002750 | -0.001787 | -0.007933 | -0.006272 | -0.013091 | -0.002775 | -0.013005 | +0.000461 | 通過 |
| `poly_quality` | +0.000123 | -0.000352 | -0.004798 | -0.001174 | -0.004095 | -0.005501 | -0.000741 | -0.001607 | 通過 |
| `poly_shift` | -0.002317 | -0.002845 | -0.018327 | -0.007557 | -0.022064 | -0.014590 | -0.012702 | -0.002413 | 失敗 |

zero-shot 顯示 `poly_quality` 的函數貼合度很好，但它不能預測 recovery 後的 class-level 穩定性。
Hardswish 不是「直接套上就保持原精度」；activation 更換會同時改變 190 個 sites 的 feature
distribution，因此需要 matched recovery。

## 8. 10-epoch short recovery

| Activation | COCO Δ | Person Δ | BBAT box Δ | BBAT pose Δ | Ball box Δ | Bat box Δ | Ball pose Δ | Bat pose Δ | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qsilu_pq` | -0.002749 | -0.000876 | -0.004285 | -0.001731 | -0.008635 | +0.000066 | -0.001702 | -0.001760 | 通過 |
| Hardswish | -0.014161 | -0.012111 | -0.011811 | -0.005198 | -0.009263 | -0.014360 | +0.006488 | -0.016884 | 失敗 |
| `poly_quality` | -0.001268 | -0.000307 | -0.012263 | -0.005681 | -0.020970 | -0.003556 | -0.006708 | -0.004655 | 失敗 |
| `poly_shift` | -0.005477 | -0.002263 | -0.016640 | -0.014514 | -0.030138 | -0.003142 | -0.020129 | -0.008900 | 失敗 |

正式 selector 的判斷不是平均 delta，而是八項最差值。這避免只看 COCO 或 overall AP 時，漏掉 BBAT5
ball／bat 的局部退化。

## 9. Finalist 與 BCSP 的誠實邊界

### Finalist

- SiLU seed-1 control：20-epoch 上限、patience 5；第 6 個已完成 epoch early-stop，選定結果 worst
  delta -0.012872，正式通過。
- qSiLU seed-1：完成 epoch 1 後進入 epoch 2 macro 106，再依使用者要求停止。epoch 1 provisional
  worst delta -0.015966，只有 BBAT ball box 超過門檻 0.000966。
- qSiLU finalist 沒有 completion marker，也沒有正式 `best_joint.pt`；不能宣稱 finalist 已失敗或勝出。

### SIPA–BCSP

- SIPA：數學、兩個 profile 的 PyTorch reference、`poly_shift` 的 Q16.10／ONNX、zero-shot、
  11-region sensitivity 與 10-epoch recovery 都有實際成果。
- BCSP：190-site manifest、11 regions、policy schema、bounded queue、dependency 與 gate propagation
  已完成。
- BCSP 實際搜尋：沒有完成。`poly_shift` prerequisite 失敗後，8 個 region recovery 與 6 個 mixed
  policies 共 14 jobs 正確標記 blocked。
- 因此目前不能宣稱 BCSP 找到最佳 placement，也不能宣稱 normalized-regret search、random／greedy
  或 ActNAS-style comparison 已完成。

## 10. 硬體與量化決策

| Activation | 結構成本判斷 | Accuracy 證據 | 決策 |
| --- | --- | --- | --- |
| SiLU | sigmoid／exp 或標準 fused op | accepted baseline | 保留量化 control |
| Hardswish | 標準 fused op 或 add/clip/multiply | recovery 未過 | 保留負結果，不進 finalist |
| ReLU | compare/max，最低成本 | zero-shot 崩潰 | 僅作 cost floor |
| qSiLU | 一個共享平方、dyadic 係數、4 個區段 | 唯一 non-SiLU recovery 通過 | 進入 matched PTQ |
| `poly_quality` | 四個資料冪乘法、一般常數 | recovery 未過 | 暫停 |
| `poly_shift` | 四個資料冪乘法、dyadic 常數 | recovery 未過 | 暫停 BCSP |

目前沒有 target FPGA／ASIC 的 synthesis、clock、precision、latency、power、LUT、DSP 或 BRAM 實測；
表中的成本是結構 proxy，不是板上量測。

量化建議固定為：

1. 使用相同 calibration images、observer、per-channel／per-tensor 與 rounding 規則，先比較 accepted
   SiLU 與完成的 qSiLU 10-epoch 權重。
2. 記錄每個 activation region 的 range、scale、zero-point、clip ratio、saturation 與 knots
   1／2／4／8 附近誤差。
3. 枚舉 Q8／Q12／Q16 全輸入碼，驗證 symmetry invariant、exact tails、overflow 與 saturation。
4. 只有 PTQ 任一八項超過 -0.015 預算時，才用 matched 10-epoch budget 同時比較 QAT SiLU 與 qSiLU。
5. 完成量化後再決定是否恢復 qSiLU 20-epoch finalist；現階段不繼續 activation-only 調參。

## 11. 可直接使用的交付

| 交付 | 路徑／SHA-256 |
| --- | --- |
| 完整八項結果 JSON | [`full35-activation-results.json`](full35-activation-results.json) |
| 八項寬表 CSV | [`full35-activation-results.csv`](full35-activation-results.csv) |
| 所有公式詳細推導 | [`full35-activation-mathematical-derivations.md`](../docs/research/full35-activation-mathematical-derivations.md) |
| Accepted SiLU 權重 | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| qSiLU 10-epoch 權重 | `7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e` |
| 權重分片與重建 | [`release/weights/README.md`](../release/weights/README.md) |

兩個權重皆已用 GitHub 分片發布並完成逐片與重建後 SHA-256 驗證。未發布 qSiLU interrupted
`last.pt`、425 MB resume checkpoints、OOM probe payload 或未過 gate 的候選權重。

## 12. 最終結論

目前真正完成並值得帶往下一階段的是：

- **SiLU**：正式 baseline、matched control 與量化對照。
- **qSiLU**：唯一通過 10-epoch 八項 gate 的 non-SiLU 候選，也是唯一應進入 matched PTQ 的新函數。
- **SIPA profiles**：數學與負結果都已完成，保留作論文消融及硬體設計參照，但目前不進一步訓練。
- **Hardswish／ReLU**：必要控制組與負結果，不是最終候選。
- **BCSP**：完成支援架構，尚未完成搜尋實驗，不能列為已驗證演算法成果。

在量化結果出來前，不需要再做 activation-only 最佳化。下一個決策點是：qSiLU 的 dyadic 分段結構
能否在相同 PTQ／QAT 規則下，保留 Full35 八項精度並降低實際部署成本。
