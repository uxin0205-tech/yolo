# 2026-09-01：MASF 精度回歸原因診斷

## 結論

現有證據不支持「MASF 固定會讓精度下降」。實際上有兩條不同問題：

1. YOLO11m P2 線把「新增 P2 graph/head」與「加入 MFAM」混在一起。P2 graph/head 先讓 validation
   mAP50-95 下降 `0.014104`；在相同 P2 graph 上加入 PaperFormula MFAM，反而回補
   `0.005914`。所以相對 B0 的 `-0.008189` 不能全部歸因給 MASF。
2. YOLO11m P3 線的舊 MFAM graft 不是 identity-safe。它在載入預訓練 parent 後立即大幅重映射
   P3 feature，再用 direct `lr0=0.01` 做全模型訓練，形成 seed-sensitive optimization shock。
   這是目前最有力且可重現的 P3 退化原因。

目前 YOLO26 已改用小係數 outer residual；同機 COCO 結果為 A0 `0.506754`、
Full35 `0.506391`、Partial75 `0.506754`。Full35 的 `-0.000362` 在 `0.001` tie band 內，
Partial75 幾乎相同，應解讀為「沒有 material gain」，不是已證明 MASF 造成下降。

## 範圍與回饋迴路

本輪只讀取既有正式 CSV、training curves、程式碼與 checkpoint，並以固定 synthetic tensor
做 CPU differential probes。沒有讀取 BBAT5 影像、沒有 GPU validation/training、沒有改動
checkpoint、資料、label 或 split。

快速回歸迴路：

~~~bash
python3 scripts/audit_accuracy_regressions.py
~~~

它會以 exit code `1` fail closed，並列出 P2 graph/head、MFAM 邊際效果與 P3 per-seed 差值。

## 一、P2 的原始歸因混入 P2 head

YOLO11m strict-fair validation mean：

| 比較 | mAP50-95 | 差值 |
|---|---:|---:|
| B0-Clean | 0.460696 | — |
| P2-Direct-Clean | 0.446593 | 相對 B0 `-0.014104` |
| P2-PaperFormula-Clean | 0.452507 | 相對 P2-Direct `+0.005914` |
| P2-PaperFormula-Clean | 0.452507 | 相對 B0 合計 `-0.008189` |

`B0 → P2-Direct` 是 P2 feature/head、正負樣本分配、隨機 P2 tower 與最佳化的合併效果；
`P2-Direct → P2-PaperFormula` 才是同 family 的 MFAM 邊際效果，而且兩個 seeds 都是正向。

原始 MASF-YOLO 是在加入 P2 後再加入 MFAM，完整系統還有 IEMA、DASI 與其他 fusion。本地只在
單一 P2/P3 neck slot 放 MFAM，並非完整論文重現；論文的 VisDrone2019、YOLO11-s 結果也不能
直接套用到 ball/bat 二類資料。[MASF-YOLO 原始論文](https://arxiv.org/abs/2504.18136)

## 二、P3 的主要根因：非 identity-safe graft

舊 strict-fair 實作為：

~~~text
Z = x + Σ DW_k(x)
y = Conv1x1(Conv1x1(Z) + x)
~~~

兩個 `1x1 Conv + BN + SiLU` 是新隨機參數，外層沒有 `y=x+g·delta` bypass。權重移植保留
原 layer 16，但新 MFAM 留為隨機初始化；因此它不是在 parent 上加小修正，而是立即改寫 P3 表示。

### 初始化差分

固定 seed 的 `2×256×80×80` synthetic feature：

| 模組 | delta / input L1 | cosine | output/input std |
|---|---:|---:|---:|
| legacy M7 | 1.057538 | 0.000328 | 0.337628 |
| PaperFormula Full | 1.056808 | -0.006366 | 0.328528 |
| PaperFormula Partial25 | 0.263360 | 0.850138 | 0.882491 |
| YOLO26 Full35 scalar residual | 0.001305 | 約 1.0 | 1.000002 |
| YOLO26 Partial75 scalar residual | 0.000323 | 約 1.0 | 1.000000 |

舊 Full 類模組一插入就有約 `105%` relative L1 改變，方向近乎正交；Partial25 仍改變全張量
約 `26%`。目前 YOLO26 的 `alpha=0.01` residual 將初始化擾動壓到千分位以下。

使用相同 official initializer、固定 `1×3×128×128` input，比較完整 B0 與 P3-Partial25 graph：

| 邊界 | relative L1 | cosine |
|---|---:|---:|
| layer 16 P3 feature | 0.301194 | 0.792314 |
| decoded prediction tensor | 0.185187 | 0.971254 |

所以擾動確實傳到偵測輸出，不只是孤立 layer 的數學現象。

### 訓練後仍大幅改寫 feature

以 `weights_only=True` 與明確 safe-global allowlist 載入正式 best checkpoints：

| 變體 | seed 42 relative L1 / cosine | seed 43 relative L1 / cosine |
|---|---:|---:|
| P3-M7 | 1.776938 / 0.081962 | 1.738507 / 0.135234 |
| P3-Lite35 | 1.614196 / 0.057371 | 1.582106 / 0.130444 |
| P3-Lite35-F7 | 1.550444 / 0.045153 | 1.574448 / 0.082738 |
| P3-Partial50 | 1.179288 / 0.232637 | 0.670841 / 0.540095 |
| P3-Partial25 | 0.313520 / 0.794630 | 0.310778 / 0.799329 |

Full 類分支持續徹底重寫 P3 feature；Partial25 的改變最小，也是 P3 family 最佳者。這支持
保留較大的 identity bypass，而不是全面多尺度混合。

## 三、最佳化與 seed 交互作用

strict-fair direct runs 都是 SGD、100 epochs、`lr0=0.01`，但新 MFAM 沒有低 LR 或
parent-preserving warmup。曲線顯示：

- P3-Lite35-F7 seed 42 在 epoch 1 最佳、31 epochs 早停；seed 43 在 epoch 53 最佳。
- P3-Partial50 seed 42 在 epoch 1 最佳、31 epochs 早停；seed 43 在 epoch 49 最佳。
- P3-Partial25 相對 B0：seed 42 `-0.032639`，seed 43 `+0.010316`。

所以平均 `-0.011161` 不是兩個 seeds 都穩定下降。只有兩個 seeds，且 B0 standard deviation
是 `0.01995`，不能把平均差宣稱成普遍結構性傷害。

P2-Control-Full 以 staged、低 LR 全模型 recovery 達 `0.492881`。它不能進 strict-fair 排名，
但足以證明訓練路徑能大幅改變結果，支持最佳化契約是重要根因。

## 四、Ball／小物件退化

P3 family 在 seed 42 的 Ball AP 下降約 `0.109–0.131`，且同時出現在 AP50、AP75、AR100 與
Ball AP_S；不是單純 NMS threshold 或高 IoU 定位問題。seed 43 多數接近持平，Partial25 甚至改善。

中等信心的機制是：

1. P3 對典型 `16×17 px` ball 只有約兩個 cells，Full 多分支再經 dense `1x1` 重映射，容易把
   稀少的局部訊號與背景 context 混在一起。
2. 多個 `Conv+BN+SiLU` branch 直接相加，沒有 branch normalization 或 learned weighting；
   trained Full checkpoints 的 feature std ratio 約 `1.45–1.66`。
3. Bat／中型物件有時改善、Ball／小物件下降，overall mAP 是 task trade-off。

本地 context-radius 實驗顯示小球需要約 `8× bbox` 的局部 context，但這只證明 context 有用，
不證明單一 P3 slot 的未加權 branch sum 是正確注入方式。可確定的是 over-mixing／feature rewrite；
「高頻被平滑」仍需真實影像 feature-spectrum probe 才能證實。

## 五、目前 YOLO26 不是同一個問題

YOLO26 使用：

~~~text
delta = Conv1x1(DW3(x) + DW5(x))
y = x + alpha * delta
~~~

`alpha=0.01` 起步，已消除舊實作最嚴重的 shock。同機 COCO internal：

| 模型 | mAP50-95 | 相對 A0 |
|---|---:|---:|
| A0 | 0.506754 | — |
| Full35 A2 | 0.506391 | -0.000362 |
| Partial75 A2 | 0.506754 | +0.000001 |

Full-data Phase B 的 Full35／Partial75 都下降約 `0.002889`／`0.002745`，更像 unfreeze
neck/detect 後的排程或 parent drift，而非 Full35 特有缺陷。A0 沒有用同一 schedule retrain，
所以仍缺 matched no-MASF control。

Full35 J3 `best_joint` 的 `p3_masf.alpha=0.110659`，代表下游 Detect/Pose 已適應此分支；
非零 alpha 不等於淨收益，也表示不能直接拔除。

後續正式 checkpoint 專項 probe 又確認：現行 in-place graft 的 layer 16 輸出會繼續進入 layer 17，
因此同時影響 P3/P4/P5；Full35 的 alpha dependency 約為 `27.1% / 17.4% / 8.5%`，Partial75 約為
`21.1% / 23.4% / 9.8%`。完整指標拆解與 Detect-only fork 解法見
[YOLO26 P3 MASF 專項診斷](<2026-09-01-yolo26-p3-masf-no-gain-diagnosis.md>)。

## 六、解決方法

### 1. 先做目前 J3 的依賴檢查

| Arm | 操作 | 目的 |
|---|---|---|
| M0 | J3 `best_joint` 原樣跑完整八項 validation | 正式 baseline |
| M1 | 同 checkpoint，只在 eval 設 `alpha=0` | MASF 即時依賴 |
| M2 | M1 通過或接近 gate 才移除 graph 並低 LR recovery | 能否安全簡化 |

若 M1 持平或改善，部署目標優先移除 MASF；若 M1 明顯下降，先保留並做 matched retraining。

### 2. 若繼續研究，改成真正 parent-preserving

~~~text
delta = Project(Σ softmax(beta_k) * DW_k(x))
y = x + gate ⊙ delta
gate = 0 at initialization
~~~

- 只保留 DW3/DW5；P3 F7/Full 沒有勝出證據。
- 先用 25% context channels／75% exact bypass。
- 比較 scalar zero gate 與 per-channel zero gate。
- branch sum 使用 normalized weights；projection 最後 BN gamma 也可 zero-init。
- 必須有 `gate=0 ⇒ output identity` 與完整 graph forward-equivalence gate。

Zero-initialized residual gate 是 ReZero 的核心；LayerScale 提供 per-channel residual scaling 先例。
兩者都應標成 BBT5 adaptation，而不是 MASF 論文原設計。
[ReZero](https://arxiv.org/abs/2003.04887)、
[LayerScale/CaiT](https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf)

### 3. 改用 staged recovery

1. Stage A：MASF/gate only，確認 gate 從零平穩離開。
2. Stage B：MASF + 相鄰 P3 neck + Detect/Pose heads；MASF LR 約 neck/head 的 2 倍，backbone 與
   BinaryQK frozen。
3. Stage C：只有 Stage B 過 gate 才用更低 LR 解凍全模型。
4. 每階段 rollback 到 accepted parent，並以相同 schedule 跑 no-MASF control。

必要時前幾 epochs 加逐步衰減的 parent feature/logit distillation：

~~~text
L = L_task + lambda_f * (1 - cosine(P3_new, P3_parent))
           + lambda_o * KD(output_new, output_parent)
~~~

它直接約束已量到的 feature rewrite，比盲目增加 epochs 更有針對性。

### 4. selection gate 要保護 Ball

- overall mAP50-95 不低於 matched control `0.001`。
- Ball AP、Ball AP_S、tiny/small recall 使用預先鎖定的退化門檻。
- Full35 同時通過 Detect/Pose、Float/Bit-True 八項 gate。
- winner 至少跑 3 個 paired seeds；兩 seed mean 不足以主張穩定提升。

### 5. P2 必須另案判斷

- control 是相同四尺度 graph 的 P2-Direct，不是三尺度 B0。
- P2 head 需要 warm-start/staged full recovery；現有 head-only stage 幾乎失敗。
- ball-only selective P2 或 auxiliary loss 是 P2/assignment 實驗，不算 MASF 修正。

## 七、最小下一輪矩陣

所有新 BBAT5 實驗只使用不可變 `bbat5-v1` assignment 與 `configs/detect.yaml`／
`configs/pose.yaml`，不得沿用歷史入口或另切 split。

| Arm | MASF | 訓練 | 目的 |
|---|---|---|---|
| C0 | 無 | matched staged schedule | 排除 retraining 效果 |
| C1 | Partial75 DW3/DW5，scalar gate=0 | 同 C0 | 最小 identity-safe 基準 |
| C2 | Partial75 DW3/DW5，per-channel gate=0 | 同 C0 | 測 channel-specific context |
| C3 | C2 + early feature/output KD | 同 C0 | 測 feature preservation |

先以固定 seed 做工程篩選；只有 winner 與 C0 跑滿至少三個 paired seeds 後才形成正式結論。
不要同時加入 Full channels、F7/F9、P2、RepConv，否則再次失去歸因能力。

## 原因排序

| 排名 | 原因 | 信心 |
|---:|---|---|
| 1 | 舊 P3 MFAM 非 identity-safe，立即重寫 pretrained P3 feature | 高 |
| 2 | direct 高 LR 與隨機新分支不匹配 | 高 |
| 3 | P2 結果混入新增 P2 graph/head 效果 | 高 |
| 4 | 單一 P3 adaptation 與論文 VisDrone/full system 不一致 | 高 |
| 5 | Full 多分支造成 Ball-sensitive over-mixing／尺度 trade-off | 中 |
| 6 | 兩 seed 方差使平均值不穩 | 高 |

最實際的下一步是先做 J3 `alpha=0` 完整 validation；若重做 MASF，優先
「Partial75 + DW3/DW5 + exact-zero per-channel residual gate + matched staged control」，
不要再重跑舊 Full PaperFormula direct100。
