# OPT-BINARYQK-SCALE-CODEBOOK：BinaryQK 少量 scale／codebook

> 2026-09-08 依[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)，本方向仍是 Q0 後的條件式，B4 無 runtime selector、A8 僅動態上界；不強制重建 FP，須先有可靠 baseline 與 Q 證據。本次禁 GPU 且沒有 replay／kernel 結果，`training_ready=false`。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

| 欄位 | 內容 |
|---|---|
| 狀態 | `proposed` |
| 全域序位 | Q1；只有 Q0 精度恢復主線仍留下 magnitude 殘差才啟動 |
| 上游方向 | [OPT-BINARYQK-ACCURACY-RECOVERY](<../binaryqk-accuracy-recovery/README.md>) |
| 現行 control | `C0-FIXED`：two-basis、per-head／basis fixed PoT |
| 部署優先候選 | `B4-GROUP`：4 個 fixed channel groups，每組 8 channels |
| 精度上界候選 | `A8-TOKEN`：per-token 8-level PoT、3-bit runtime index |
| 首輪工作 | cached Q/K replay、operation trace、target-kernel gate；不先開長訓練 |
| 架構圖 | [目前、B4 與 A8 資料流](<architecture-report.md>) |
| 執行方式 | [條件式最小計畫](<plan.md>) |
| 證據入口 | [證據與數值索引](<evidence.md>) |

## 一句話決策

「準備 8 個 scale」本身不會自動補精度；關鍵是 **每個 activation 要使用哪一個 scale，以及這個
assignment 是離線固定還是每張圖動態決定**。

- 若部署禁止 per-image magnitude reduction／selector，先測 `B4-GROUP`。它沒有 runtime
  scale 選擇，但必須證明 4-way partial popcount 沒有吃掉 BinaryQK 的速度收益。
- 若要判斷 token magnitude 是否真是剩餘精度瓶頸，保留 `A8-TOKEN` 作 fidelity ceiling。它每張
  圖仍要重算 token magnitude、產生 3-bit index，並做 pair-level exponent／shift。
- 不優先 global 4/8-entry codebook。完整 global dynamic 已只比 fixed PoT 高
  `0.000505537` mAP，僅關閉 FP–PoT 缺口的 `4.5767%`。
- 目前沒有 A8／B4 detection mAP 或 target-kernel 證據，不能宣稱它們可補回約 `0.011`。

## 它在整體優化流程的位置

~~~text
最終 same-lineage FP winner
          │
          ▼
Q0 BinaryQK：fixed-PoT site isolation
          │
          ▼
matched low-LR QAT → 必要時 FP-teacher ranking KD／STE-window
          │
          ├─ gap 已可接受 ─────────────────────────────> 停止，不開本方向
          │
          └─ gap 仍明顯，且 probe 指向 magnitude/scale 殘差
                                   │
                                   ▼
                   Q1 本方向：C0／B4／A8 cached replay
                                   │
                                   ▼
                      target kernel + bit-true gate
                                   │
                                   ▼
                    最多一個 candidate 進完整 validation/QAT
~~~

因此本資料夾是[BinaryQK 精度恢復主線](<../binaryqk-accuracy-recovery/README.md>)的條件式子方向，不是
新的第一順位。site 10／site 22 的敏感度與 sign/ranking adaptation 尚未釐清前，先增加 scale
granularity 會把「位置、量化與硬體」混成同一個變因。

## 目前 scale 到底怎麼做

正式 fixed-PoT 設定在 calibration 時，先由 Q/K 的
`[B,H,D,N]` activation 求 per-image／head／basis global magnitude，再把觀察到的 combined
coefficient 平均、量化並凍結：

~~~text
calibration：
Q/K [B,H,D,N]
      │
      └─> mean(abs(.), axes=D,N)
                 │
                 └─> combined coefficient
                            │
                            └─> calibration mean → PoT round → c_fixed[h,basis]

正式 inference：
packed Q/K bits → XNOR-popcount → 乘 c_fixed[h,basis] → basis sum
~~~

兩個 sites × 4 heads × 2 Hadamard bases，共有 `16` 個固定 combined-coefficient slots。A-FINAL
實際只使用 `0.125` 與 `0.25`。因此演算法上的 fixed inference **不需要每張圖片重算 scale**；
scale 是校正後保存的常數。

目前 PyTorch eager reference 仍有一個工程冗餘：fixed mode 在分支判斷前會先算 dynamic
`abs → mean`，再把結果丟掉。這應以 early-return 修掉並做 bit-true parity，但它只改善實作與
profiling，不會改變輸出，也不是精度恢復方法。

## 三個候選的真正差異

| ID | scale assignment | 每張圖 reduction／selector | 主要代價 | 角色 |
|---|---|---:|---|---|
| `C0-FIXED` | 每 site/head/basis 離線固定 | 無 | 16 個 fixed slots | 現行 control |
| `B4-GROUP` | 每 8 channels 一組，離線固定 | 無 | 4-way partial popcount／shift-add | 部署優先候選 |
| `A8-TOKEN` | 每 token 動態選 8 個 PoT level之一 | 有 | reduction、index traffic、variable shift | fidelity ceiling |

在 `S=2, B=2, H=4, N=400, D=32` 下：

- C0 two-basis 有 `2,560,000` 個 full-word basis terms。
- B4 有 `64` 個 fixed combined-coefficient slots，但 naïve partial terms 上界增加到
  `10,240,000`。
- A8 每張圖需要 `409,600` magnitude abs、`396,800` reduction adds、`12,800` 個 3-bit
  indices（最小 `4,800 B`），以及最多 `2,560,000` 個 pair exponent／variable shifts。

這些是 analytical operation／storage 上界，不是 cycle、latency 或能耗實測。完整公式與
`1/sqrt(32)` 的 PoT fold 限制見[架構圖](<architecture-report.md>)及
[研究報告](<../../docs/research/2026-09-03-binaryqk-scale-codebook-hardware.md>)。

## 本方向只做什麼

納入：

- 重用同一 parent、同一 cached Q/K 與相同 validation samples，比較 C0、B4、A8。
- 同時量 score fidelity、index／scale 統計、bit-true parity 與 target-device kernel 成本。
- 只從 B4／A8 中帶一個 Pareto winner進完整 detection validation。
- 完整 validation 過 `+0.001` AP gate 後，才做 candidate／C0 matched QAT。

首輪不納入：

- global K=4/8 codebook、K=4 token codebook或 G=8 channel groups。
- 同時改 basis、threshold、normalization、site policy、KD loss或資料 split。
- 只憑 BitOPs／Python reference 時間宣稱硬體加速。
- 把 Full35 多任務 metrics與較早的單任務 YOLO26 scale ablation混成同一 lineage。
- 承諾補回全部約 `0.011` mAP。

## 文件與來源

- [條件式最小實驗計畫](<plan.md>)
- [目前、B4、A8 的 terminal 架構圖與公式](<architecture-report.md>)
- [本地證據、推導數值與未證明事項](<evidence.md>)
- [完整第一手研究報告](<../../docs/research/2026-09-03-binaryqk-scale-codebook-hardware.md>)
- [2026-09-03 原始決策紀錄](<../../docs/worklogs/2026-09-03-binaryqk-scale-codebook-decision.md>)
- [2026-09-04 資料夾整理紀錄](<../../docs/worklogs/2026-09-04-binaryqk-scale-codebook-folder-organization.md>)

實驗開始後才新增 `results.md`；原始 metrics、trace、checkpoint與大型 artifacts仍留在所屬實驗
目錄，本方向只保存可追溯的決策摘要與連結。

返回[優化方向索引](<../README.md>)。
