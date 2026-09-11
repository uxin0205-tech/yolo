# OPT-BINARYQK-ACCURACY-RECOVERY：BinaryQK 精度恢復

> 2026-09-08 本輪依[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)：先比較 J3 `best_joint`／J3 `best_pose`／J2 `best_joint` 選 PSEL，必要基準修復後才做 BinaryQK 單因子診斷／獨立 challenger；原 J3 `best_joint` 只作 `B0` 歷史分數參照。Q/K baseline `qk_ste=false` guard 仍明確禁止 `STE=true`，不說直接改 YAML 可跑；Full35 Float 不等於 FP-QK，新 parent／site screens 需重做。`training_ready=false`，GPU 0。舊 S6/S7 與單任務 W-DIR LR 內容保留為歷史，詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

| 欄位 | 內容 |
|---|---|
| 狀態 | `proposed` |
| 全域序位 | Q0；先完成並凍結最終 FP 架構 |
| 目前缺口 | YOLO26 A-FINAL 相對 B26-FP 為 `-0.011641` mAP50-95 |
| 首選實用候選 | 保留 fixed PoT，site isolation 後只二值化較不敏感的 attention site |
| 首輪新工作 | 2 個 validation-only site jobs；通過才做 2 個 matched QAT jobs |
| 條件式工作 | 單一 FP-teacher KD／STE-window；per-token dynamic 只作 accuracy ceiling |
| 不在首輪做 | progressive、normalization sweep、BDCN、dual-basis、threshold sweep |
| 架構圖 | [目前、候選與修改後資料流](<architecture-report.md>) |
| 執行方式 | [最小實驗計畫](<plan.md>) |

## 方向結論

BinaryQK 不是沒有救回精度的可能。舊 YOLO11 單-site 實驗曾由 raw sign 的 `-0.051554`，經
scaled sign、QAT 與 feature/attention KD 縮到 `-0.000830`；但現行 YOLO26 是兩個 sites，W-DIR
做過 full-model direct recovery 後仍差 `0.010540`，A-FINAL 仍差 `0.011641`。不能把舊模型的成功
直接外推，也不應再把普通 full-model recovery 當成新方法。

納入硬體「不能每張圖重算大量 scale」的需求，以及既有消融後，最實用的順序是：

1. 直接重用已完成的 YOLO26 `dynamic / static-head / PoT` scale消融，不再重跑。
2. 保留正式 fixed PoT，完整驗證 `site10 binary only` 與 `site22 binary only`；敏感 site 保留 FP。
3. 只對 site winner 做 direct QAT；仍有 ranking/KL gap才加一種 FP-teacher loss。
4. saturation 仍高時才測 positive pre-sign scale／learnable STE window；它不改 inference sign bits。
5. per-token dynamic 會要求每張圖重算 scale與 pairwise epilogue，只保留為條件式 accuracy ceiling。
6. 目前 sign ratio
   沒有嚴重偏斜，所以 threshold 不是第一順位。

這個方向是**主 FP 訓練完成後的 QAT/recovery optimization**，仍需要訓練；不是 PTQ 後處理。
正式 production parent 必須是完成 MASF、RepConv 等架構決策後的同 lineage FP winner。

## 現在為什麼會掉精度

目前正式 V1-BR／A-FINAL 的 `scale_mode` 是 `power_of_two`。程式在 calibration 時先用
Q/K layout `[B,H,D,N]` 算 global dynamic coefficient：

~~~text
mu_q[b,h] = mean_{d,i} |q[b,h,d,i]|
mu_k[b,h] = mean_{d,j} |k[b,h,d,j]|

c_dyn[b,h,basis]
  = gamma[basis] · mu_q[b,h] · mu_k[b,h] / sqrt(D)

c_fixed[h,basis]
  = PoT_round(mean_calibration c_dyn[:,h,basis])

S_current[i,j]
  = c_fixed[h,identity] · z_identity[i,j]
  + c_fixed[h,hadamard] · z_hadamard[i,j]
  + relative_bias[i,j]
~~~

因此 calibration 使用每 sample/head 的 global magnitude，但正式 inference 使用固定的
`[1,H,1,1]` coefficient；同一 head/basis 的所有影像、query token與 key token共用同一數值。A-FINAL
checkpoint 的實際值為：

| site | head 0 | head 1 | head 2 | head 3 |
|---|---|---|---|---|
| `model.10` identity / Hadamard | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 |
| `model.22` identity / Hadamard | 0.125 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 |

正式 Full35 同樣設定為 `basis: hadamard`、`scale_mode: power_of_two`；兩個 sites × 4 heads × 2 bases
合計只需保存 16 個離線常數，不必為每張 image重新估計。

所以從演算法與硬體資料流來看，正式推論應是：

~~~text
calibration（只做一次） ─> 儲存 0.125 / 0.25 ─> 每張圖直接用 bit shift
~~~

不過目前 PyTorch reference 的 `_coefficient()` 有一個可修的軟體冗餘：它先呼叫
`_dynamic_coefficient()` 計算每張圖的 `abs → mean`，之後 fixed mode才回傳 `_fixed()`，前者不會
影響輸出。現行 Hadamard 有兩個 bases，所以每個 site 白算 Q/K/Q_h/K_h 四次 magnitude reduction，
兩個 sites共八次。這不代表 fixed PoT 硬體必須重算；正式 profiling／export 前應改成：

~~~text
目前 eager：dynamic reduction ─> 判斷 fixed ─> 丟棄 dynamic ─> 取 fixed coefficient
應改控制流：若 fixed 且已校正 ─> 直接取 fixed coefficient
             否則             ─> dynamic reduction／calibration
~~~

這項 early-return 是 latency／乾淨實作修正，不是精度恢復實驗；本輪只記錄問題，未修改
`yolo_attention` production code。

令 `q_i = mu_q b_qi + r_qi`、`k_j = mu_k b_kj + r_kj`，則：

~~~text
q_i^T k_j
  = mu_q mu_k b_qi^T b_kj
  + mu_q b_qi^T r_kj
  + mu_k r_qi^T b_kj
  + r_qi^T r_kj
~~~

identity／Hadamard 各自都用 sign-dot 加一個共用 fixed coefficient近似；後三個 residual terms 仍會
隨 query/key token pair 改變，不是乘一個 fixed coefficient、`gamma` 或調 softmax temperature 就能
完整恢復。結果會同時出現：

- 每個 token 原本不同的 magnitude 被壓成一個 head-level 常數。
- pairwise score 差值與 top-k 排序改變，attention map 變平。
- 兩個 sites 同時二值化；site 10 的誤差還會繼續傳入 Neck 與 site 22。
- clipped STE 只讓 `|x| <= 1` 通過 sign gradient，部分 heads 有約 35%–54% 元素落在飽和區。

相反地，Exact／PWL／SHIFT normalization 的本地 zero-train 差異都遠小於 `0.001`，所以目前約
`0.011` 的 gap 不是 normalization approximation 造成。

## 你先前已完成的 scale／magnitude 消融

你的記憶正確，不應重複做已回答過的實驗。

YOLO26 在同一 W-DIR parent 上已完成三個 evaluation-only scale screens：

| Run | scale | 推論是否每張圖重算 | mAP50-95 | 相對 dynamic |
|---|---|---:|---:|---:|
| V1-DYN | per-sample/head global dynamic | 是 | 0.507457 | 0 |
| V1-SHEAD | calibrated fixed per-head | 否 | 0.506879 | -0.000579 |
| V1-P2 | calibrated fixed per-head/basis PoT | 否 | 0.506952 | -0.000506 |

PoT 只少 `0.000506`，在既定 `0.001` tie band內，因此選硬體更簡單的 V1-P2是合理決策。這表示
`DYN-GLOBAL` 不需要重跑，也表示 global dynamic 的每-image reduction目前不值得優先帶回 production。

舊 YOLO11 也做過相關但不完全相同的消融：

- E1-S sign-only `0.45930` → E1 global scaled-sign `0.48056`：dynamic global magnitude對未訓練替換
  幫助很大，但仍比 FP 低約 `0.03029`。
- T1 sign QAT `0.50789` → T2 global scaled-sign QAT `0.50951`：回補約 `0.00162`。
- T3/T4/T5 dual-basis內部已有 per-token channel-mean scales，但同時改了 residual basis、threshold與
  QK products，不能單獨歸因 per-token scale。
- N4-FP/I8/I4 加過 per-token magnitude rank-1 side channel；最佳 N4-I4 `0.50986`，未超過 T7-D
  `0.51165`。N4 是「額外加 `mq_i·mk_j`」，不是「用 `mq_i·mk_j` 乘 binary dot」，公式不同。

YOLO26 T5-SCR 也使用 per-token residual-basis scales，結果 `0.492907`，未勝 Hadamard H-SCR
`0.495243`；同樣因 basis與 scale一起改，不能當成乾淨的 per-token 單因子結果。

## 條件式 accuracy ceiling：per-token/head magnitude

只改 scale 軸，不先更換 sign、basis、bias、normalization 或 sites：

~~~text
mu_q[b,h,i] = mean_d |q[b,h,d,i]|
mu_k[b,h,j] = mean_d |k[b,h,d,j]|

S_token[i,j]
  = gamma · mu_q[b,h,i] · mu_k[b,h,j]
    · <sign(q_i), sign(k_j)> / sqrt(D)
~~~

兩張 COCO image 的 CPU 機理 probe，比較的是 `global dynamic` 與 `per-token dynamic`，在兩個
YOLO26 sites 都同方向改善：

| site | global top-10 overlap | per-token top-10 overlap | global KL | per-token KL |
|---|---:|---:|---:|---:|
| `model.10.m.0.attn` | 0.4206 | **0.5119** | 0.5431 | **0.4651** |
| `model.22.m.0.1.attn` | 0.4962 | **0.5952** | 0.8810 | **0.6188** |

這只證明 score fidelity 候選值得保留，**不是 mAP 證據**，也不是和正式 fixed-PoT A-FINAL 的直接
比較。既有 scale ablation 中 global dynamic `0.507457`、PoT `0.506952`，只差 `0.000506`；因此
PoT 固定化不是整個 `0.011` gap 的主因。只有前面的 fixed-PoT site isolation、direct QAT、KD／
STE-window仍失敗，而且確定願意研究 dynamic-scale hardware時，才成對執行 `DYN-GLOBAL` 與
`TOKEN-BOTH`，以兩者差值歸因 token granularity。per-token 不是照抄 BinaryAttention 官方實作；
官方採 global magnitude，本方案是依本地 token dispersion 提出的候選。

部署成本也必須誠實計入。以 `B=1,H=4,N=400`、兩個 sites 計算，共用一套 token scale時，Q/K
需 `6,400` 個 token scales，兩個 `N×N` score epilogue約 `1,280,000` 個 pair-scale operations/image；
這是單一 scale組的下限。若 identity與 Hadamard各自保留獨立 scale，最壞會加倍到 `12,800` 與
`2,560,000`。若無 tiled fusion、PoT或其他有效 epilogue，這些運算可能吃掉 binary matmul的部分
收益。因此 per-token 是 accuracy-ceiling 候選，不是目前的 deployment-first winner。

## 已獨立的少量 scale／codebook 子方向

「準備8個 scale」的問題已拆成獨立的
[OPT-BINARYQK-SCALE-CODEBOOK](<../binaryqk-scale-codebook/README.md>)，避免和本方向的 site policy、
QAT及 KD主線混成同一個實驗。

目前結論只保留三點：

- `C0-FIXED` 是現行 control；16個 per-site/head/basis fixed slots不需每張圖重算。
- `B4-GROUP` 是無 runtime selector的 deployment-first候選，但 naïve partial-popcount上界為
  `10.24M`，必須先過 target-kernel gate。
- `A8-TOKEN` 是 per-token magnitude fidelity ceiling，每張圖仍要 reduction、3-bit index與
  variable shift，不是「只存8個常數」。

這個 P2子方向只有在 fixed-PoT site isolation、matched QAT、單一 KD／STE-window後仍有
magnitude殘差才啟動。完整的[條件式計畫](<../binaryqk-scale-codebook/plan.md>)、
[目前／候選架構圖](<../binaryqk-scale-codebook/architecture-report.md>)與
[證據索引](<../binaryqk-scale-codebook/evidence.md>)集中維護於該資料夾。

## 第一個新實驗：site isolation／hybrid precision

現行兩個 binary sites：

- `model.10.m.0.attn`：較上游，誤差會進入後續 Neck。
- `model.22.m.0.1.attn`：靠近 P5 Detect feature。

兩張影像的 FP-parent CPU output probe 顯示，任一 site 單獨二值化都會造成明顯 decoded-output
drift；site 10 的 drift 也會傳到 site 22。這仍不能判定哪個 site 的 AP 更敏感，所以需要兩次
完整 validation：

| 候選 | site 10 | site 22 | 用途 |
|---|---|---|---|
| `ISO-10-BIN` | Binary | FP | 量 site 10 單獨代價 |
| `ISO-22-BIN` | FP | Binary | 量 site 22 單獨代價 |

若只有一處特別敏感，保留它為 FP、只把另一處二值化，通常會比增加 basis 更直接。代價是 binary
coverage 下降；是否值得必須看 target-device end-to-end latency，不能只看 binary-op proxy。

## 第二階段：單一 FP-teacher loss + direct QAT

若 hybrid winner 已改善但仍未接近 matched FP control，才加一個 teacher arm：

- attention probability KL 高但 ranking 尚可：選 attention-map KL。
- top-k／pairwise ranking 仍亂：選 ranking-aware loss。
- 不同時疊 output、feature、attention-map、ranking 四種 losses。

teacher 只存在於訓練，不進 deployment graph。舊 YOLO11 的 KD 結果與 BinaryAttention、Bi-ViT
原始研究支持這個候選，但不能代替新的 YOLO26 matched experiment。完整來源與外推限制見
[第一手證據報告](<../../docs/research/2026-09-02-binaryqk-accuracy-recovery-primary-sources.md>)。

## checkpoint 與 lineage 限制

- W-DIR 的 `0.507457` 是既有最佳 accuracy 數字，但其 `best.pt` 已永久清理；不能 resume。
- W-DIR 本來就是全模型解凍、`lr=5e-5` 的 direct recovery；再做 generic full-model recovery 不是
  新實驗。
- 目前 retained、可做低成本診斷的 binary checkpoint 是 V1-BR，mAP `0.506658`：
  `/home/uxin/yolo/yolo_attention/artifacts/runs/v1-br/ultralytics/weights/best.pt`。
- 從 V1-BR 把一個 site 還原 FP，只能回答「這個已共同適應的模型能否成為 hybrid」，不能當作
  乾淨的 single-site 因果證明。
- 正式 QAT 必須從最終同 lineage FP winner 重新建立；不能把舊 COCO lineage、MASF/RepConv 變體
  或不同 evaluator 的 checkpoint 混排。

## 本方向首輪範圍

納入：

- V1-BR 上只新增 2 個 validation-only screens：`ISO-10-BIN`、`ISO-22-BIN`；既有 V1-P2/V1-BR
  fixed-PoT metrics與已完成的 V1-DYN/SHEAD/P2 scale ablation直接重用。
- 最佳候選與 matched FP control 的 direct QAT，各一個 seed。
- 通過後才補 paired seeds；必要時只加一個 teacher KD arm。
- mAP、AP_S/M/L、關鍵類別、score fidelity、binary coverage、target latency 與 memory gate。

不納入：

- 原樣重跑 W-PROG 或只延長 W-DIR。
- PWL／SHIFT／BDCN normalization 搜尋。
- full residual dual／multi-basis；它會增加 2–4 倍 binary QK products。
- 沒有 sign imbalance 證據時做 threshold sweep。
- 首輪 per-token dynamic；只有 fixed-PoT/hybrid recovery失敗且另行接受 runtime scale成本才開啟。
- 同時加入 MASF、RepConv 或改資料 split。
- 沒有 custom binary kernel、bit-true parity 與 target profile 就宣稱加速。

執行順序、停止條件與結果契約見[最小實驗計畫](<plan.md>)；終端可讀的原本／預計／條件式架構
見[架構圖報告](<architecture-report.md>)。本次整理紀錄見
[中文工作紀錄](<../../docs/worklogs/2026-09-02-binaryqk-accuracy-recovery-direction.md>)。

返回[優化方向索引](<../README.md>)。
