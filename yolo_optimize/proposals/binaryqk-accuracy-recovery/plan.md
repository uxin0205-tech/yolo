# BinaryQK 精度恢復：最小必要實驗計畫

> 2026-09-08 總計畫S6/S7為準；Full35 Float不等於FP-QK，新parent重做site screens，Q/K hardware_frozen需核對；單任務W-DIR LR不能搬成joint已驗證recipe。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

本計畫只回答一個問題：在保留有意義的 1-bit QK 覆蓋率與部署收益下，能否把現行約 `0.011`
mAP50-95 缺口補回。原因與證據見[方向 README](<README.md>)，原本／候選資料流見
[終端架構圖](<architecture-report.md>)。

## 一、放在整體流程的哪裡

~~~text
完成 MASF、train-time RepConv 等 FP 架構決策
                    │
                    ▼
選出並凍結 final same-lineage FP winner
                    │
                    ├─ 現在可先用 retained V1-BR 做免訓練診斷
                    │
                    ▼
本方向：BinaryQK site/scale 選擇 + matched direct QAT
                    │
                    ▼
選出並凍結 FP／hybrid／both-binary accuracy winner
                    │
                    ▼
RepConv deployment fuse（若有）→ bit-true/export
                    │
                    ▼
INT8 PTQ/QAT、representative calibration、binary kernel、target profile
~~~

「主模型訓練完成後」不代表本方向免訓練；BinaryQK 的 direct QAT 是小範圍 recovery training。
若 RepConv 還是 trainable architecture change，必須先固定；只有等價的 deployment fusion 放在
BinaryQK winner 後。PTQ/calibration 不能先做，因為它不會恢復 sign 已丟掉的 Q/K 資訊。

## 二、兩種 parent，各自只回答一種問題

### A. 立即診斷 parent

使用 retained V1-BR：

~~~text
/home/uxin/yolo/yolo_attention/artifacts/runs/v1-br/ultralytics/weights/best.pt
mAP50-95 = 0.506658
~~~

它只用於兩個免訓練 site-isolation validations。還原單一 FP site 的結果屬 deployment-oriented
hybrid 診斷，不作完整因果宣稱。已完成的 V1-DYN／V1-SHEAD／V1-P2 scale screens直接重用。

### B. 正式訓練 parent

使用所有上游 FP 架構決策完成後的 final same-lineage FP winner：

- 同一 dataset、split、evaluator、head mode 與程式版本。
- 尚未做 BinaryQK、PTQ、INT8 calibration 或 deployment fuse。
- 保存 checkpoint digest、config snapshot、commit 與完整 FP metrics。

W-DIR checkpoint 已刪除，不能當 parent；A-FINAL 是 queue policy winner，不是最高 accuracy parent。

## 三、Phase 0：先過工程與診斷 gate

沒有 GPU training 前，先完成以下檢查：

1. 稽核既有 V1-DYN／V1-SHEAD／V1-P2 metrics、parent與 evaluation-only狀態；不重跑 scale screen。
2. `fixed-PoT` reference 與現行 V1-BR output 一致；正式 checkpoint 的 coefficients必須被保留。
3. fixed mode 必須在 dynamic magnitude reduction 前 early-return；bit-true output需與現行 reference一致，
   避免把 eager reference「先算後丟」的冗餘成本帶入 target profiling或 export。
4. site policy 只能改 `model.10.m.0.attn`、`model.22.m.0.1.attn`，其他 state tensors 保留。
5. deterministic sign 的 zero policy、XNOR-popcount 與 float sign-dot bit-true 一致。
6. Float save/reload、deep-copy best checkpoint、materialization/export 都保留 scale/site policy。
7. 對固定 validation samples 保存每 site/head 的 score cosine、NRMSE、top-k overlap、KL、entropy、
   sign ratio、STE saturation 與 downstream feature drift。

目前兩張 COCO CPU per-token probe 只作條件式 accuracy-ceiling 證據，不影響首輪 fixed-PoT
site-isolation；若未來開啟該支線，再保存固定 sample IDs 的 machine-readable 結果。

## 四、Phase 1：只做兩個免訓練完整 validation

以同一個 V1-BR checkpoint、同一 evaluator 執行：

| ID | site 10 | site 22 | scale | 新訓練 | 回答問題 |
|---|---|---|---|---:|---|
| `FIXED-P2-BOTH` | Binary | Binary | fixed per-head/basis PoT | 0 | 既有 V1-BR baseline，可重用正式 metrics |
| `ISO-10-BIN` | Binary | FP | 現行 fixed PoT | 0 | 只保留 site 10 為 1-bit 是否較好 |
| `ISO-22-BIN` | FP | Binary | 現行 fixed PoT | 0 | 只保留 site 22 為 1-bit 是否較好 |

因此只有 2 個新 validation jobs，不加 training；`FIXED-P2-BOTH` 重用既有正式 metrics。兩個新
候選都要報：

- overall AP50-95、AP50、AP75、AP_S/M/L。
- sports-ball、baseball-bat 或正式任務的關鍵類別。
- 每個 site 的 score fidelity、feature drift、attention entropy。
- 1-bit QK 覆蓋率與 analytical cost；不能把 hybrid 稱為全 BinaryQK。

### Phase 1 選擇規則

1. 若一個 hybrid 相對 `FIXED-P2-BOTH` 明顯提高 AP，正式候選採該 site policy；`+0.001` 視為
   material improvement。
2. 若兩個 hybrid 都未改善，但其中一個退化明顯較小，可把它帶入一次 matched QAT；首 seed未過即停。
3. 兩個 hybrid 都比 baseline 低超過 `0.001` 且 binary coverage/latency也沒有價值：停止，不啟動 QAT。
4. 每輪最多只帶一個 candidate，不把 per-token、threshold、KD 同時堆疊。

## 五、Phase 2：兩個 matched direct-QAT jobs

從正式 final FP parent 建立同 seed 的兩個 arms：

| Arm | 架構 | Trainable | 用途 |
|---|---|---|---|
| `FP-CTRL` | 保持 FP QK | 與候選相同的全模型 scope | 排除普通 fine-tune gain |
| `BQK-CAND` | Phase 1 唯一 winner | 全模型 + 必要 quantizer params | 量 BinaryQK 的真正 paired gap |

第一輪沿用既有 W-DIR 的 direct protocol 作固定起點：epoch 0 即使用目標 binary score、AdamW、
`lr=5e-5`、最多 40 epochs、patience 5；兩 arms 的資料、augmentation、optimizer、scheduler、AMP、
seed、evaluator 與 stopping rule 完全一致。W-DIR 已證明 full-model direct 比本地 progressive 更好，
所以不再加入 staged unfreeze 或 score blending 這些第二變因。

### Phase 2 gate

同時滿足才進下一步：

- `BQK-CAND - FP-CTRL` 的 AP50-95 gap 理想值不低於 `-0.001`。
- 至少要比現行 YOLO26 `-0.010540` recovery gap 縮小 `0.001` 以上，否則不算 material recovery。
- AP_S/M/L 與關鍵類別沒有用一個類別犧牲換 overall。
- 無 NaN/Inf、sign collapse、異常 entropy 或 state-loss。
- analytical binary coverage 合理；target-device latency 尚未量到前只標 `accuracy candidate`。

首個 seed 未過：停止，不調 LR/epoch、不掃 gamma、不加 KD。首個 seed 通過：先依下一節判斷是否
需要 KD；選出最終候選後再補兩組 paired seeds，使 `FP-CTRL` 與最終 BinaryQK arm 各 3 seeds。

~~~text
Phase 1：2 validation-only jobs
Phase 2 初篩：2 training jobs
沒有通過：總訓練成本停在 2 jobs
通過後：只對最後候選補 paired seeds
~~~

## 六、Phase 3：只有 ranking/KL 明確失敗才加一個 KD arm

`BQK-CAND` 已有 improvement、但 paired gap 仍大於 `0.001` 時，從同一 FP parent 新增一個
`BQK-CAND-KD`。teacher 是 frozen `FP-CTRL` parent，teacher 不進 deployment graph。

只能依預先診斷選一種 loss：

| 診斷 | 唯一新增 loss |
|---|---|
| ranking 尚可、attention probability KL/entropy 差 | temperature-aware attention-map KL |
| top-k overlap／pairwise ranking 明顯差 | ranking-aware attention loss |

KD arm 的 schedule、trainable scope 與 `BQK-CAND` 相同。它相對 no-KD 必須提升 `≥ +0.001` 且
轉化成 detection AP，否則停止。不要因為 KD 失敗就依序再疊 feature/output/attention 四種 loss。

## 七、Phase 4：只保留一個診斷式備案

若 fixed-PoT hybrid + 單一 KD 後仍有 gap，且正式 probe 重現大量 `|Q|>1`／`|K|>1`：

~~~text
x_scaled = x / softplus(c_h)
b = sign(x_scaled) = sign(x)       # c_h > 0，forward bits 不變
d b / d x 使用可學 head-wise STE window
~~~

這個 positive pre-sign scale 只改 QAT gradient window，正值下不改 inference sign bits，理論上可在
部署時移除；仍需測 state/export 等價。只有 sign ratio 明顯偏離 50% 時才考慮 threshold `tau_h`；
本次 probe 約在 44%–57%，不支持先做 threshold sweep。

full residual dual-basis 只作最後 accuracy ceiling，因它可能把 binary QK products由一次增到四次，
不列入本最小計畫。

## 八、條件式少量 scale／codebook 子方向

只有前述 fixed-PoT site isolation、direct QAT、單一 KD／STE-window仍失敗，且正式診斷仍指向
magnitude／scale殘差時，才移交
[OPT-BINARYQK-SCALE-CODEBOOK](<../binaryqk-scale-codebook/README.md>)。本主線不重跑既有 global
dynamic／PoT消融，也不在此維護另一份 C0／B4／A8矩陣。

子方向先做 cached Q/K replay與 target-kernel gate，不先開長訓練；最多只帶一個 B4或A8 winner
回到完整 validation／matched QAT。完整順序、計數與停止條件見
[條件式最小計畫](<../binaryqk-scale-codebook/plan.md>)。

## 九、資料、結果與部署契約

- 診斷使用 V1-BR 原本同一份 COCO validation與 evaluator，不改 protocol。
- 若正式模型使用 BBAT5，新 detection 實驗只能使用不可變的
  `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與 `configs/detect.yaml`；不得重切、抽樣或改標註。
- 每個 job 保存 config、seed、commit、parent digest、trainable names、best/last digest、完整 metrics、
  score diagnostics 與 failure reason。
- 結果摘要寫入本方向 `results.md`；checkpoint 與原始 artifacts 留在正式實驗目錄。
- per-token scale 需另量 scale storage、pair-scale epilogue、p50/p95 latency、throughput、peak memory。
- PyTorch `sign()`／fake quant 不是 1-bit runtime。必須有 bit packing、binary QK kernel、bit-true
  parity、export artifact 與 target-device profile，才能宣稱加速。
- INT8 P/V、Conv/Linear 與 1-bit QK 是不同 quantization boundaries，需分開校正與驗證。

## 十、最後決策

| 結果 | 動作 |
|---|---|
| Phase 0 工程 gate 失敗 | 修正 implementation，不啟動 validation/training |
| Phase 1 兩個 hybrid 都失敗 | `rejected`，保留 FP／現行 BinaryQK，不花 QAT 成本 |
| Phase 2 首 seed 未過 | `rejected`，停止 scale/site 方向 |
| Phase 2 通過但 KD 無 material gain | 保留 no-KD candidate，不再疊 loss |
| 3 paired seeds 過 accuracy gate、但 target latency 不過 | 保留為研究結果，production 採 FP/hybrid |
| accuracy、bit-true 與 target profile 全過 | `validated`，凍結 BinaryQK winner 後再進 PTQ/calibration/export |

返回[方向說明](<README.md>)、[架構圖](<architecture-report.md>)或[優化方向索引](<../README.md>)。
