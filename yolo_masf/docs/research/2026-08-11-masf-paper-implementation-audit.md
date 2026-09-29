# MASF-YOLO 論文與現有 MFAM 實作稽核

日期：2026-08-11
範圍：只稽核 MFAM、P2/P3 placement、partial-channel/F7 adaptation、zero-init residual gate 與訓練設定。
第一手來源：[`field_check/MASF-YOLO An Improved YOLOv11 Network for Small Object.pdf`](../../field_check/MASF-YOLO%20An%20Improved%20YOLOv11%20Network%20for%20Small%20Object.pdf)，共 6 頁。

## 結論摘要

1. 論文原版 MFAM 不是只放在 P2 或 P3。Figure 1 在 backbone 畫出四個 MFAM；新增 P2 detection layer 是另一項獨立改造。（論文 III Proposed Method，PDF p.2；Figure 1 與 III-A，PDF p.3）
2. 論文原版 MFAM 包含四個卷積分支：`DW3×3`、`DW5×5`、`DW1×7 → DW7×1`、`DW1×9 → DW9×1`，並加入 identity `X`。沒有 learnable branch weights，也沒有 learnable gate。（論文 III-A、Figure 2、式 (1)–(6)，PDF p.3）
3. 論文文字／公式與 Figure 2 對尾端 projection/residual 有內部不一致：Figure 2 視覺上是五路相加後接一個 `Conv1×1`；式 (6) 則是 `W = Conv1×1(Conv1×1(Z) ⊕ X)`，也就是兩個 1×1，中間另有一次 `X` residual。若要求「依論文公式」，應以式 (6) 實作並在報告揭露 Figure 2 的不一致。
4. 現有 [`masf_yolo/models/mfam.py`](../../masf_yolo/models/mfam.py) 忠實實作平行 3/5/7/9 分支、identity 加總及最後一個 1×1 fusion，但**沒有式 (6) 的內層 `Conv1×1(Z) ⊕ X` 與第二個 1×1**。它較接近 Figure 2 的簡圖，不是完整的公式重現。
5. P2-only、P3-only、Partial50、Partial25、F7-only，以及 zero-init residual gate 都是本棒球專案的 adaptation，不能標成 MASF-YOLO 論文原版。
6. 論文訓練設定只有：VisDrone2019、SGD、initial LR 0.01、momentum 0.937、cosine annealing、batch 12、100 epochs、640×640，以及其軟硬體環境。論文沒有說 pretrained/random init、freeze、warmup、weight decay、augmentation、seed 或 patience。（論文 IV-A/B，PDF p.4）因此本 repo 的 B0 初始化、10+90 freeze schedule 和較低 LR 都是本研究訓練策略，不是論文設定的完全重現。

## 1. 論文原版 MFAM

### 1.1 確切位置

論文先把兩項改造分開描述：

- 新增 P2 small-object detection layer，保留較高解析度的淺層資訊。
- 在 backbone 使用 MFAM，讓 backbone 擷取多尺度 context。

這不是「P2 head 內建 MFAM」的同一項設計。III Proposed Method 明說 MFAM 幫助 **backbone network** 擷取 context；III-A 再次說 MFAM 增強 backbone 對小物件的 feature extraction。（PDF p.2–3）

Figure 1（PDF p.3）畫出的 backbone 流程可讀為：

```text
Input
  → CBS → CBS → C3k2 → MFAM
  → CBS → C3k2 → MFAM
  → CBS → C3k2 → MFAM
  → CBS → C3k2 → SPPF → C2PSA → MFAM
```

也就是圖上共四個 MFAM，位於四個 backbone stage 的輸出附近；最後一個在 `C2PSA` 後。把這四處推定為 P2/P3/P4/P5 backbone stages 是根據 Figure 1 的尺度序列所作的架構判讀，論文文字沒有逐一用「P2、P3、P4、P5」標示四個 MFAM，因此報告時應標成「由 Figure 1 推定」，不可寫成作者逐字明述。

### 1.2 分支與公式

依 III-A、式 (1)–(6)（PDF p.3），四個卷積輸出為：

```text
Y1 = DWConv3×3(X)
Y2 = DWConv5×5(X)
Y3 = DWConv7×1(DWConv1×7(X))
Y4 = DWConv9×1(DWConv1×9(X))
```

融合為：

```text
Z = Y1 ⊕ Y2 ⊕ Y3 ⊕ Y4 ⊕ X
W = Conv1×1(Conv1×1(Z) ⊕ X)
```

論文明定 `⊕` 是 element-wise addition、`X` 是 input、`Z` 是四個多尺度 feature 與 input 的逐元素和、`W` 是 MFAM output。故可確定：

- 四個卷積分支彼此平行，不是依序串接 3→5→7→9。
- identity `X` 與四個分支等權相加；不是 concat。
- 論文沒有 branch attention、softmax 權重、scale parameter 或 gate。
- 7 與 9 使用 `1×k` 後 `k×1` 的 factorized strip convolutions。
- 相較 PKI Module，作者拿掉 11×11，並把 3×3 改成平行，動機是降計算量並避免串接造成小物件語意流失。（III-A，PDF p.3）

### 1.3 論文自身的結構圖／公式矛盾

Figure 2（PDF p.3）的可見資料流是：四個 convolution branches 加上 identity，五路 Add 後接一個 `Conv1×1` 到 output。可是同頁式 (6) 明確給出：

```text
W = Conv1×1(Conv1×1(Z) ⊕ X)
```

這表示公式版本有兩個 1×1，且兩者之間還有 `X` residual。兩者不能同時完全成立。稽核採以下口徑：

- 「paper-formula MFAM」：以式 (1)–(6) 為準，使用兩個 1×1 與中間 residual。
- 「paper-figure MFAM」：依 Figure 2，五路相加後只使用一個 1×1。
- 若論文重現實驗只能選一個，優先使用明確的數學式版本，並在報告註記 Figure 2 不一致。

論文 III-A 結尾把 MFAM 寫成「MKAM」，依上下文判斷是排版筆誤，不能據此推導成另一個模組。

## 2. 論文訓練與評估設定

### 2.1 論文明述的設定

IV-A/B（PDF p.4）提供：

| 項目 | 論文設定 |
|---|---|
| Dataset | VisDrone2019 |
| Framework | PyTorch |
| OS / CUDA | Ubuntu 20.04 / CUDA 11.3 |
| GPU | NVIDIA GeForce RTX 4090D 24 GB |
| Optimizer | SGD |
| Initial LR | 0.01 |
| Momentum | 0.937 |
| LR schedule | cosine annealing |
| Batch size | 12 |
| Epochs | 100 |
| Input | resize 至 640×640 |

論文**沒有交代** pretrained 或 random initialization、freeze schedule、warmup、weight decay、augmentation、AMP、seed、deterministic、early stopping/patience、gradient accumulation、exact Ultralytics commit/version。缺少的項目不能由 repo 設定反推成「論文做法」。

### 2.2 論文消融能證明什麼

Table I（PDF p.5）以 YOLOv11-s 做累積消融：

| 累積架構 | mAP50 | mAP50:95 | Params |
|---|---:|---:|---:|
| Baseline | 0.446 | 0.294 | 9.42 M |
| + P2 | 0.468 | 0.307 | 9.62 M |
| + P2 + MFAM | 0.478 | 0.319 | 10.93 M |
| 再 + Fusion | 0.483 | 0.321 | 11.00 M |
| 再 + IEMA | 0.488 | 0.324 | 11.01 M |
| 再 + DASI（MASF-YOLO） | 0.492 | 0.329 | 12.05 M |

這是**累積消融**，只支持「在已加 P2 的模型上再加入論文配置的 MFAM 後，該實驗指標提高」，不能拆出每個 MFAM placement、每條 kernel branch 或 residual 寫法各自的因果效果。Table II（PDF p.5）報告 MASF-YOLO-s validation/test mAP50:95 為 0.329/0.268，YOLOv11-s 為 0.294/0.240；這些是 VisDrone 結果，不能直接當 BBT5 的預期門檻。

IV-C（PDF p.4–5）的指標是 P、R、mAP@0.5、mAP@0.5:0.95、Params、GFLOPs。論文沒有 AP_S/AP_M/AP_L，也沒有 Ball/Bat per-class AP；本專案新增這些指標是合理的任務導向評估，但必須標成 adaptation。

## 3. 現有 repo 實作對照

### 3.1 忠實之處

[`masf_yolo/models/mfam.py`](../../masf_yolo/models/mfam.py) 的 `MFAM` 有以下忠實部分：

- `kernels=(3,5,7,9)` 的 full variant 與論文四個尺度一致。
- 3/5 使用 depthwise grouped spatial convolution。
- 7/9 使用 `1×k → k×1` 串接，而且兩個大尺度分支與其他分支平行。
- forward 從 `summed = value` 開始，再將每個 `branch(value)` 逐元素相加，忠實做到 `Z = X ⊕ ΣYi`。
- 最後以一個 1×1 `fuse` 輸出，與 Figure 2 的簡圖一致。

[`masf_yolo/variants.py`](../../masf_yolo/variants.py) 的 Phase 1 `M0=(3,5,7,9)` 是 repo 中 branch set 最接近論文原版的變體。

### 3.2 偏離或無法證明忠實之處

| 項目 | 論文 | 現有實作 | 判定 |
|---|---|---|---|
| 尾端公式 | `Conv1×1(Conv1×1(Z) ⊕ X)` | `fuse(Z)`，只有一個 1×1 | 符合 Figure 2，但偏離式 (6) |
| MFAM placement | Figure 1 backbone 共四處 | Phase 1 主要放 P2 slot；`P3M` 只放 P3 slot | placement adaptation |
| full branch set | 3/5/F7/F9 | 只有 M0 使用全 branch；其他為消融 | M0 branch set 忠實，其他是 adaptation |
| learnable gate | 無 | 現行 `mfam.py` 無；新規格提議加入 | 現行在此點忠實；新 gate 偏離 |
| DWConv 細節 | 文字稱 depthwise separable convolution，但未公布 BN/activation/order | Ultralytics `Conv(..., g=channels)`，每個 Conv 自帶該版本的 normalization/activation | 無法宣稱逐算子完全重現 |
| 訓練 | batch 12、LR 0.01、100 epochs 等 | BBT5、batch 16、B0 initializer、分階段與較低 LR | task adaptation，不是 paper reproduction |

特別注意：一般術語「depthwise separable convolution」常包含 depthwise spatial convolution 加 pointwise convolution；現有 branch 是 `groups=channels` 的 spatial Conv，整個模組最後才有共用 1×1。由於論文同時用 `DWConv` 符號、Figure 2 又沒有揭露 BN/activation/逐分支 pointwise，單靠六頁論文無法判定作者實際 code。應寫「依公開公式可對應」，不要寫「逐 layer 與作者 code 完全相同」。

### 3.3 現有 slot 的實際位置

[`configs/models/yolo11m-p2-slots.yaml`](../../configs/models/yolo11m-p2-slots.yaml) 在 P2 neck `C3k2` 後保留 P2 slot，並在下採樣、Concat、P3 `C3k2` 後保留 P3 slot；Detect 接收 `[P2, P3, P4, P5]`。[`masf_yolo/models/builder.py`](../../masf_yolo/models/builder.py) 將兩個 slot 固定為 module index 20/24、channels 128/256，並依 variant 安裝 `MFAM`、`PartialMFAM` 或 identity。

因此 Phase 1 的 P2/P3 MFAM 位於 **neck/detection feature slot**，不是論文 Figure 1 所畫的四個 backbone placements。這是為 BBT5 小球與硬體成本設計的刻意 adaptation，不是錯誤，但命名與報告必須講清楚。

## 4. 使用者指定的 adaptations

### 4.1 P2-only 與 P3-only

新規格 [`docs/superpowers/specs/2026-08-11-b1r-p2-p3-retest-design.md`](../superpowers/specs/2026-08-11-b1r-p2-p3-retest-design.md) 定義：

- P2 family：以 validation 選出的 revised P2 parent 為基礎，MFAM 只放 P2。
- P3 family：以原始 B0 三尺度權重為 parent，MFAM 只放 P3，維持 P3/P4/P5 detection。

這兩個 family 都不是論文四處 backbone MFAM 的重現。它們回答的是「在 BBT5 上，把同一配方限制於一個高解析 feature level 是否值得」，屬於硬體友善 placement ablation。跨 P2/P3 family 的 parent 不同，只能做 operational comparison；嚴格因果比較應在各 family 內相對各自 parent 判讀。

### 4.2 Partial50 與 Partial25

現有 `PartialMFAM` 把 channel 切成 processed 與 bypass：

```text
processed 50% 或 25% → MFAM
其餘 50% 或 75%    → exact identity
最後 channel concat
```

這符合 [`MFAM_plan.md`](../../MFAM_plan.md) 中 M2/M3 的本研究定義，也符合現有程式的 split/process/concat，但論文沒有 partial-channel MFAM。故 `P2/P3-Partial50-35`、`P2/P3-Partial25-35` 是本研究 adaptation；不可宣稱來自 MASF-YOLO。

### 4.3 F7-only

`P2/P3-Full-35-F7` 使用 `DW3`、`DW5`、`DW1×7→DW7×1`，移除 F9。它保留論文的平行與 factorized-kernel 思想，但少了原版 `DW1×9→DW9×1`，因此是使用者指定的硬體友善 branch ablation，不是 paper-full MFAM。

只有 `P2/P3-Full-35-F7-F9` 的 **branch set** 等同論文原版；但因為 placement 只在 P2 或 P3、尾端 residual 仍需選定公式或圖版本、訓練資料與設定不同，它仍不能被稱為「完整 MASF-YOLO 重現」。

若「SU3」是指目前討論中的 P3 25% partial 變體，應清楚命名為 `P3-Partial25-35`：只在 P3、25% channels 經 3/5 branches、75% exact bypass。若「SU3」實際指另一個 variant，必須先在 variant contract 中另行定義，不能用未定義縮寫宣稱符合論文。

## 5. residual zero gate 判定

新規格提議：

```text
output = input + gate × mfam_delta(input)
gate 為 per-channel learnable parameter，初始化為 0
```

判定：**這明確偏離論文。**

- 論文式 (5) 的 identity 是 `Z` 內的固定等權加法。
- 論文式 (6) 雖有 `Conv1×1(Z) ⊕ X` residual，但沒有乘法 gate、沒有 per-channel learnable coefficient，也沒有 zero initialization。
- 提議的 outer identity bypass 在 gate=0 時讓整個新模組精確等於 input；論文 MFAM 初始化時沒有此保證。

此 gate 的合理研究目的，是在 B0/B1R parent 上加入新模組時先保持 parent function，降低 random MFAM 立即破壞已訓練 feature 的風險。它可以作為「identity-preserving fine-tuning adaptation」，但不得命名成 paper residual，也不得以它的結果宣稱重現論文 MFAM。

如果實驗目標同時包含「忠實論文」與「盡量貼近 B0」，最清楚的處理是拆成兩個明確 contract：

1. `PaperFormulaMFAM`：四個原版 branches，依式 (5)–(6)，不含 learnable gate；用來回答論文公式重現。
2. `IdentityPreservingMFAM`：同樣的 branch ablation，但使用 zero-init per-channel gate；用來回答 BBT5/B0-preserving fine-tuning。

兩者不可混成同一名稱。若 GPU 預算只允許一條主線，而目前首要目標是避免 B0 能力被新模組破壞，zero-gated adaptation 可作主線；報告必須把它標成 adaptation，另以 CPU 結構測試證明 paper-formula 版本的公式契約，或保留至少一個不帶 gate 的 paper-formula 對照。

## 6. 對後續實作與報告的明確要求

1. 不應寫「P2/P3 MFAM 完全照論文」；應寫「kernel branches 取自 MASF-YOLO，placement/partial/gate 為 BBT5 adaptation」。
2. 要宣稱符合論文式 (6)，實作必須具有 `Z → Conv1×1 → Add X → Conv1×1 → W`；現有單一 `fuse(Z)` 不足。
3. 要宣稱符合論文 full branch set，必須包含 3/5/F7/F9；F7-only 是消融。
4. Partial50/25 必須報告 processed/bypass channels 與 exact bypass；不可歸因為論文原模組。
5. 若未來另做 zero-gate variant，必須在名稱、config manifest、checkpoint metadata 與報告中標示，並有 `gate=0 ⇒ output≈input` 的單元測試；本輪 paper-formula 主線已決定不使用 gate。
6. 訓練報告要分開列「paper-reported settings」與「repo actual settings」。100 epochs、SGD、momentum 0.937、cosine、640 可說沿用論文；batch 16、B0 data-exposed initializer、10+90 freeze、staged LR 0.01/0.001、direct LR 0.001、AMP/seed/augmentation 都是 repo 設定。
7. 結果需列 overall 與 Ball/Bat 的 mAP/AP，以及 AP_S/AP_M/AP_L；但要註明後兩類細分不是論文原始評估。COCO size AP 與 baseball-specific size buckets 不可混名。
8. 論文 Table I 是累積消融，不能拿來主張「只在 P3」或「Partial25」必定有相同提升；P2/P3 M0–M4 必須以本次 BBT5 validation/test 實證。

## 來源定位

- MASF-YOLO 論文：III Proposed Method（PDF p.2–3）、III-A MFAM 與 Figure 1/2、式 (1)–(6)（PDF p.3）、IV-A/B/C（PDF p.4–5）、Table I/II（PDF p.5）。
- 研究計畫：[`MFAM_plan.md`](../../MFAM_plan.md)，特別是 B1、M0–M3、S0 與正式訓練規則。
- 現行 MFAM：[`masf_yolo/models/mfam.py`](../../masf_yolo/models/mfam.py)。
- 現行 placement/variant：[`masf_yolo/models/builder.py`](../../masf_yolo/models/builder.py)、[`masf_yolo/variants.py`](../../masf_yolo/variants.py)、[`configs/models/yolo11m-p2-slots.yaml`](../../configs/models/yolo11m-p2-slots.yaml)。
- 新一輪 adaptation 規格：[`docs/superpowers/specs/2026-08-11-b1r-p2-p3-retest-design.md`](../superpowers/specs/2026-08-11-b1r-p2-p3-retest-design.md)。
