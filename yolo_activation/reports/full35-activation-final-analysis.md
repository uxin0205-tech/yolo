# Full35 Activation 最終整合分析與量化交接

> **文件狀態：2026-08-29 歷史詳細分析。** 表格與當時停止點保留作追溯；目前完成邊界、
> 可讀數學與下一步以[權威整合報告](completed-activation-integrated-report.md)為準。


## 結論先行

本階段可以收尾並交給量化工作列，但不能宣稱 activation 最終勝負已經由 20-epoch finalist 完整證明。
可以凍結的事實如下：

1. 所有實驗從 `yolo_combine/final/full35` accepted J3 `best_joint` 接手；不是從隨機初始化訓練。
2. 使用完整 COCO2017 與不可變 Canonical BBAT5 v1，`fraction=1.0`、不重切、不抽樣、沒有 30% 資料夾。
3. 完成 baseline、全 train profiling、五種 uniform zero-shot、11-region `poly_shift` sensitivity、四種
   10-epoch recovery、SiLU finalist control，以及 qSiLU finalist 的一個 provisional epoch。
4. 10-epoch hard gate 中，唯一八項全過的非 SiLU 候選是 `qsilu_pq`；Hardswish、`poly_shift`、
   `poly_quality` 均失敗。故「Hardswish 比 qSiLU 好」不是本次正式結果。
5. `poly_shift` prerequisite 失敗後，8 個 region recovery 與 6 個 mixed policy 共 14 jobs 按預註冊
   dependency 正確封鎖，不應為了填滿表格繼續浪費訓練。
6. qSiLU 20-epoch finalist 依使用者要求在 epoch 1 完成、epoch 2 macro 106 後停止，不能把其
   `last.pt` 稱為 finalist 或量化父權重。
7. 量化應先比較 accepted SiLU 與已完成 10-epoch qSiLU 兩個權重。activation 與 quantization 確實強耦合：
   dynamic range、負谷、分段門檻、導數、rounding 與 saturation 會共同決定 PTQ／QAT 誤差。

## 資料與模型血緣

| 項目 | 凍結值 |
| --- | --- |
| 架構 | YOLO26m Full35 shared trunk + COCO80 Detect + BBAT5 Pose26 |
| 父權重 | `yolo_combine/final/full35/weights/combined/inference/best_joint.pt` |
| 父權重 SHA-256 | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| 父權重來源 | J3 global epoch 58 best joint；訓練在 global epoch 63 early-stop |
| COCO2017 | 118,287 train／5,000 val；COCO80 Detect |
| BBAT5 | `/home/uxin/yolo/original/pose/derived/bbat5-v1/`；5,964 formal train／683 formal val |
| 資料比例 | `1.0`；`resampling=false` |
| 影像尺寸 | 640 |
| Activation sites | 190 個 references，分 11 regions |
| Selector | Bit-True；八項 mAP50-95 各自相對 accepted baseline 比較 |
| Hard gate | 每一項 delta 都必須 `>= -0.015`；不平均 COCO 與 BBAT raw AP |

COCO 與 BBAT5 是同一個 shared-trunk run 的不同 task head。BBAT5 box 來自 Pose26 head 的 box branch，
COCO 指標來自 COCO80 Detect head；沒有把 BBAT5 二類 Detect YAML 錯接到 COCO80 head。

## 原始 Full35 與 activation recovery 配置

原父權重的 stage lineage：

| Stage | Epoch 上限 | Patience | Warm-up | 主要可訓練範圍／LR |
| --- | ---: | ---: | ---: | --- |
| J0 | 8 | 0 | 1 | Pose head `2e-4` |
| J1 | 20 | 8 | 1 | heads `2e-4`、neck `7.5e-5` |
| J2 | 80 | 17 | 1 | heads `2e-4`、MASF `1.5e-4`、neck `7.5e-5`、backbone `1.5e-5` |
| J3 | 20 | 5 | 3 | heads `5e-5`、MASF `3.8e-5`、neck `1.9e-5`、backbone `3.8e-6`、attention `5e-7` |

Activation 實驗不是把每個候選重新跑 128 epochs，而是從同一個 accepted inference EMA state
乾淨載入，重新建立 optimizer、scheduler、EMA 與 RNG：

| Phase | Epoch | Seed | Patience／warm-up | LR scale | 實際狀態 |
| --- | ---: | ---: | --- | ---: | --- |
| Baseline reproduction | 0 | 0 | — | — | 完成 |
| Full train profiling | 0 | 1 | — | — | 完成，COCO 118,287／BBAT5 5,964 全看完 |
| Uniform zero-shot | 0 | 1 | — | — | 5 activations 完成 |
| `poly_shift` region zero-shot | 0 | 1 | — | — | 11/11 完成 |
| Uniform short recovery | 10 | 1 | 0／1 | J3 `0.1×` | 4 候選完成 |
| Region recovery／BCSP | 10 | 1 | 0／1 | J3 `0.1×` | 14 jobs 因 prerequisite 失敗封鎖 |
| SiLU finalist control | 最多 20 | 1 | 5／3 | J3 `1.0×` | epoch 6 early-stop，完成並過 gate |
| qSiLU finalist | 最多 20 | 1 | 5／3 | J3 `1.0×` | 人工停止；只有 epoch 1 provisional |
| Seed 2／ablation | 20 | 2／1 | 5／3 | J3 `1.0×` | 未啓動 |
| PTQ | 0 | 1 | calibration | — | 交給量化工作列 |
| QAT if needed | 10 | 1 | 0／1 | J3 `0.5×` | 只在 PTQ 超預算時啓動 |

所有 recovery 共用 AdamW、`weight_decay=2.7e-4`、betas `(0.948,0.999)`、AMP、gradient clip 10、
cosine final factor 0.5。10-epoch LR 為 backbone `3.8e-7`、neck `1.9e-6`、MASF `3.8e-6`、
attention `5e-8`、兩個 heads `5e-6`；finalist 使用上表原 J3 LR。

## Batch 與記憶體事實

- Detect logical batch 固定 128，每 macro 兩個 logical batches，共 256 Detect images；這才是 optimizer
  看到的有效 batch 語義。
- qSiLU／多項式 recovery 的 physical microbatch 為 16，以 16 次 physical forward 組成 macro；
  Pose physical batch 16、每 macro 一批、loss weight 0.25。
- SiLU finalist physical microbatch 32；validation 固定 Detect 32／Pose 16。
- qSiLU physical batch 128、64、32 都在第一個 forward OOM，尚未進入 backward；穩定完成訓練的是 16。
  因此目前 31.35 GiB GPU 上不能在「不做梯度累積」的前提下使用 32／64／128。
- 這是 PyTorch eager reference 的顯存結論，不等於未來 fused qSiLU kernel 的上限。

## Accepted baseline

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

## Uniform zero-shot 結果

| Activation | Worst delta | Gate | 解釋 |
| --- | ---: | --- | --- |
| `qsilu_pq` | -0.013091 | 通過 | 未訓練替換已在八項預算內 |
| `poly_quality` | -0.005501 | 通過 | zero-shot 最貼近，但 recovery 不穩定 |
| `poly_shift` | -0.022064 | 失敗 | BBAT overall／ball box 較敏感 |
| Hardswish | -0.176230 | 失敗 | 直接替換造成大幅 feature drift |
| ReLU | -0.947526 | 失敗 | 八項均為 0，僅作為診斷下界 |

Hardswish 起初看起來「某些數值較高」的混淆來自不同階段、不同指標或修復前異常 run；在同一 accepted
父權重、同一完整資料、同一 Bit-True 八項口徑下，zero-shot 並不高。它經 10 epochs 可大幅恢復，
但 bat pose 仍超過 gate。

## 10-epoch short recovery：正式 selector

| Activation | COCO | BBAT box | BBAT pose | Worst delta | Gate／失敗項 |
| --- | ---: | ---: | ---: | ---: | --- |
| `qsilu_pq` | 0.495273 | 0.625751 | 0.901986 | -0.008635 | 通過；八項全過 |
| Hardswish | 0.483861 | 0.618224 | 0.898519 | -0.016884 | 失敗；BBAT bat pose |
| `poly_quality` | 0.496755 | 0.617772 | 0.898036 | -0.020970 | 失敗；BBAT ball box |
| `poly_shift` | 0.492545 | 0.613396 | 0.889203 | -0.030138 | 失敗；BBAT box、ball box、ball pose |

這張表說明：只看 COCO 會誤選 `poly_quality`，只看 overall 指標也會漏掉 ball／bat class-level 風險。
八項逐項 gate 正是為了保護 COCO + BBAT5 的共同父模型。

## `poly_shift` 11-region sensitivity

| Region | Sites | COCO delta | BBAT box delta | BBAT pose delta |
| --- | ---: | ---: | ---: | ---: |
| backbone attention | 3 | -0.000228 | -0.000775 | +0.000571 |
| backbone deep | 21 | -0.001191 | -0.001247 | -0.001273 |
| backbone early | 21 | +0.000255 | +0.000465 | -0.002520 |
| detect one-to-many | 18 | 0 | 0 | 0 |
| detect one-to-one | 18 | -0.001220 | 0 | 0 |
| MASF | 3 | -0.000006 | +0.000202 | -0.000332 |
| neck | 33 | -0.000934 | -0.003098 | -0.007104 |
| neck attention | 1 | -0.000075 | -0.000045 | -0.000001 |
| pose flow | 24 | 0 | 0 | 0 |
| pose one-to-many | 24 | 0 | 0 | 0 |
| pose one-to-one | 24 | 0 | -0.012352 | -0.000141 |

zero-shot 顯示 attention／MASF 較低風險、neck 與 pose one-to-one 較敏感；但預註冊規則要求
`poly_shift` uniform 10-epoch prerequisite 先通過。它沒有通過，所以這些觀察不能被包裝成已訓練的
mixed policy 成果。

## Finalist 狀態

SiLU seed-1 control 使用 20-epoch 上限、patience 5，在第 6 個已完成 epoch early-stop；選中的
gate-passing epoch 八項最差 delta 為 `-0.012872`，正式完成。

qSiLU 使用相同 seed／epoch／LR budget，但 physical microbatch 16。停止前只完成 epoch 1：COCO
`0.494248`、BBAT box `0.620522`、BBAT pose `0.897755`；唯一未過的是 BBAT ball box，delta
`-0.015966`，只超門檻 `0.000966`。它隨後進入 epoch 2 macro 106，才依使用者要求中止。
沒有 completion marker、沒有 `best_joint.pt`，所以此數值只能寫 provisional，不能拿來宣稱 qSiLU
finalist 失敗或獲勝。

## 數學與硬體定位

六個函數的逐式推導見[完整數學文件](../docs/research/full35-activation-mathematical-derivations.md)。
簡要定位：

- `poly_quality`／`poly_shift` 屬於 SIPA 受約束積分多項式族，精確保持
  `A(x)-A(-x)=x`、zero anchor、負谷、`C²` 與 exact ReLU tails。
- `qsilu_pq` 使用 `|x|=1,2,4,8`、dyadic 系數的 `C¹` 分段二次偶殘差；一個共享平方，尾端精確 ReLU。
- qSiLU 是 hardware-friendly 設計候選，但目前只有函數、Q16.10、ONNX 與 Full35 accuracy 證據；沒有
  指定 FPGA／ASIC 的 synthesis、latency、power、LUT/DSP/BRAM 量測，不能說已經實現硬體加速。
- qSiLU 名稱與分段二次 SiLU 近似都已有相關 prior art；可主張本專案凍結候選與證據，不能主張廣義方法新穎。

## 權重發行為何只放兩個

GitHub 一般單 blob 上限為 100 MB，而每個 inference 權重是 106,825,541 bytes。本次以 90,000,000-byte
以下分片發布，並附重建與 SHA-256 檢查：

1. accepted SiLU `best_joint`：量化 baseline／正式父權重。
2. 完成 10-epoch 且過 gate 的 qSiLU `best_joint`：唯一可交付的 non-SiLU quantization candidate。

沒有上傳 425 MB optimizer resume checkpoint、29 GB runs、OOM probe payload、未完成 qSiLU finalist
`last.pt`，也沒有上傳已被 gate 淘汰候選的權重。這樣保留量化所需的最小充分對照，不把 Git 歷史變成
不可維護的 artifact 倉庫。

## 給量化工作列的建議順序

1. 重建並校驗兩個發布權重；禁止使用中止的 qSiLU finalist `last.pt` 當正式父權重。
2. 同一 calibration split、同一 observer、同一 per-channel／per-tensor 規則分別跑 SiLU 與 qSiLU PTQ。
3. 除網絡 AP 外，逐 activation region 記錄 scale、zero-point、clip ratio、saturation、負谷 code、
   門檻 1/2/4/8 附近誤差，以及式 `A(n)-A(-n)=n` 是否 bit-exact。
4. Q8／Q12／Q16 枚舉全輸入碼，比較 floor／round-to-nearest／symmetric rounding、overflow 與 saturation。
5. 只有 PTQ 任一八項超出 `-0.015` activation budget，才對 SiLU 與 qSiLU 使用相同 10-epoch QAT budget；
   不應只替失敗者補訓。
6. 完成 PTQ/QAT 後再決定是否恢復 qSiLU 20-epoch finalist；目前繼續做 activation-only 最佳化的收益
   小於先釐清 activation × quantization interaction。

## 可追溯產物與限制

- 完整 selector 與 full Float／Bit-True zero-shot 數值：
  [`full35-activation-results.json`](full35-activation-results.json)。
- 八項寬表：[`full35-activation-results.csv`](full35-activation-results.csv)。
- 凍結 recipe：[`training/full35/activation-recipe.yaml`](../training/full35/activation-recipe.yaml)。
- 原始 29 GB artifacts 留在本機、受 `.gitignore` 排除且未刪除。
- 無指定 target board／clock／precision，板上速度與功耗仍是未解事項。
- qSiLU finalist 未完成、seed 2 未執行；任何多 seed 最終優越性主張都必須等待量化決策後補齊。
