# 2026-09-03 poly_shift PTQ、特殊格式與 QAT 入口報告

## 結論

`poly_shift + LSQ+ A8` 已完成從 W8、W7、W6、W5、W4 的逐區敏感度、mixed-bit、Fixed-SD4、Paper-TWN 與耦合 PTQ。現階段最穩健的 PTQ 主線仍是九個 deployment region 使用 W8、`backbone_attention_safe` 保留 FP32；若只把一個 Detect predictor path 改成 Fixed-SD4，最差 total mAP50 由 `-0.011855` 改善為 `-0.011415`，最差 total mAP50–95 維持 `-0.017322`，總 deployment weight storage 約 `3.551×`。

較高壓縮的 Detect head Fixed-SD4 候選約 `3.611×`，但最差 mAP50–95 已到 `-0.039788`，距 `-0.04` 只剩 `0.000212`；三條特殊 route 的硬體候選約 `3.616×`，mAP50／mAP50–95 分別只剩 `0.000786`／`0.000723` 餘裕。因此它們只能進 matched QAT／LS-SD4 recovery，不應直接宣稱 final winner。

Paper-TWN 在本輪 PTQ 未通過雙門檻：balanced route 為 `recover`，safe route 為 `reject`。它保留作三元權重研究對照，但目前沒有理由優先於 W4 或 SD4 進正式結果。

## 指標契約

- 主硬門檻：activation 替換與 weight 量化合計後，八項 mAP50 對 accepted Full35 的每一項 delta 都必須 `>= -0.015`。
- 伴隨門檻：同樣八項 mAP50–95 的每一項 delta 都必須 `>= -0.04`。兩族都通過才是 `green`。
- 八項各包含 COCO aggregate box、COCO person、BBAT aggregate box／pose、ball box／pose、bat box／pose。平均值不能蓋過最差 task。
- COCO box、BBAT box 與 BBAT pose 的 aggregate 是 evaluator 官方輸出；跨 Detect／Pose 的 `mean_required` 與 `joint_priority` 只作研究診斷／checkpoint 排序，不能稱為官方單一 mAP。
- `joint_priority` 固定為 COCO box `0.2`、COCO person `0.2`、BBAT box `0.2`、BBAT pose `0.4`，mAP50 與 mAP50–95 各自計算。

accepted Full35 的八項診斷平均 mAP50／mAP50–95 為 `0.921146／0.819760`，joint priority 為 `0.887985／0.781683`。只加入 recovered `poly_shift + A8` 的 matched parent 後，分別為 `0.923427／0.816646` 與 `0.888455／0.777968`；也就是 activation 對 joint mAP50 約 `+0.000470`，對 joint mAP50–95 約 `-0.003714`。後續所有 total delta 都已包含這段 activation 差異。

## 逐區 bit sensitivity

| 位寬 | 實際 search 格數 | dual green | recover | reject | 邊界結論 |
|---|---:|---:|---:|---:|---|
| W8 | 累積十區 | 九區 policy green | 十區含 backbone attention recover | 0 | `backbone_attention_safe` 保留 FP32 |
| W7 exact | 10 | 8 | 2 | 0 | backbone early、backbone attention 不直接晉級 |
| W6 exact | 8 | 7 | 1 | 0 | backbone deep 只進 recovery |
| W5 exact | 7 | 5 | 1 | 1 | neck recover；Detect tower reject |
| W4 exact | 5 | 4 | 0 | 1 | MASF、neck attention、Pose tower／predictor 可用；Detect predictor reject |

這些結果證明低位寬不能按 backbone→head 單調外推。例如 Detect predictor 的 isolated W5 通過，但 W4 嚴重失敗；Fixed-SD4 在該 path 卻能通過。選 route 必須看真實任務指標，而非只看參數量或 weight MSE。

## mixed-bit 結果

| 候選 | 最差 total mAP50 | 最差 total mAP50–95 | 判定 |
|---|---:|---:|---|
| small W4 frontier | -0.012838 | -0.019397 | green，但只量化少量區域 |
| heads frontier | -0.016345 | -0.040675 | recover |
| nine-region quality | -0.018478 | -0.038258 | recover |
| nine-region balanced | -0.018478 | -0.041035 | recover |
| nine-region boundary | -0.029315 | -0.054582 | recover，僅作高壓縮 QAT 候選 |

九區 W8 仍是 PTQ Pareto 主線：packed deployment weight 為 `25,437,120` bytes、約 `3.549×`，最差 mAP50／mAP50–95 為 `-0.011855／-0.017322`。

## 特殊格式 isolated 結果

- neck attention：exact W4 (`-0.009279／-0.013014`) 稍優於 Fixed-SD4 (`-0.010064／-0.013240`)。
- Detect tower：Fixed-SD4 為 green (`-0.009940／-0.037822`)；同 paths exact W4 因 mAP50–95 `-0.042992` 只列 recover。
- Detect predictor：Fixed-SD4 為 green (`-0.008289／-0.012978`)；同 path exact W4 為 reject (`-0.114854／-0.129161`)。
- Paper-TWN balanced 為 recover (`-0.037325／-0.074978`)；Paper-TWN safe 為 reject (`-0.064370／-0.111516`)。三元權重不進第一輪 QAT 主線。

## 九區 W8 與特殊 route 耦合

| 候選 | 總 packed bytes | 壓縮比 | 最差 mAP50 | 最差 mAP50–95 | joint Δ50 | joint Δ50–95 | dual |
|---|---:|---:|---:|---:|---:|---:|---|
| + SD4 predictor | 25,426,880 | 3.551× | -0.011415 | -0.017322 | -0.000609 | -0.006567 | green |
| + W4 neck attention | 25,404,352 | 3.554× | -0.013343 | -0.018451 | -0.001462 | -0.007274 | green |
| + W4 neck attention + SD4 predictor | 25,394,112 | 3.555× | -0.013020 | -0.018451 | -0.001268 | -0.007427 | green |
| + SD4 tower | 25,013,184 | 3.610× | -0.013116 | -0.040012 | -0.001276 | -0.015182 | recover |
| + SD4 Detect head | 25,002,944 | 3.611× | -0.012750 | -0.039788 | -0.001026 | -0.015217 | green |
| + W4 neck attention + SD4 tower | 24,980,416 | 3.614× | -0.014499 | -0.038690 | -0.001842 | -0.015445 | green |
| 三條特殊 route | 24,970,176 | 3.616× | -0.014214 | -0.039277 | -0.001663 | -0.015683 | green |

雙指標後的三個角色：

1. accuracy：九區 W8＋SD4 Detect predictor。
2. balanced／LS-SD4 研究：九區 W8＋SD4 Detect tower＋predictor；必須用 QAT 拉開 mAP50–95 安全距離。
3. hardware extreme：再加入 W4 neck attention；只保留探索角色，不作目前 final 推薦。

## QAT 入口狀態

- CPU graph preflight：148 個 W8 weight quantizer、124 個 LSQ+ A8 quantizer、396 個 quantizer optimizer parameters、6 個 qparam groups、4 個 Binary Q/K protected modules、final BN 為 0。
- RTX 5090 smoke：physical Detect batch 32 會 OOM；改為 16 後通過。logical Detect batch 維持 128，以梯度累積實現；Pose physical batch 16。
- 通過 smoke 的峰值顯存為 `26,541 MiB`，396/396 qparam 皆有有限梯度，gradient norm before clip `1.2901`。
- matched sham 使用 AdamW、J3 的 `0.1×` role LR、3 epoch warmup、前5 epoch scale-only、總15 epoch、不加額外 noise。實際 loader 為 COCO train 與 hash-pinned BBAT5 search train；validation 為 COCO val 與固定 BBAT5 search-val，不碰 formal val。
- QAT arm 已加入 fail-closed 規則：沒有完成、通過 gate 且 checkpoint SHA-256 相符的 matched-sham `best_joint.pt`，真正 QAT 不得啟動。

## 還需要完成

1. 完成 V19 matched sham；確認每項 mAP50／mAP50–95、相對 matched parent 絕對漂移 `<= 0.01`，並產生 hash-pinned `best_joint`。
2. 只有第1項通過才跑 V19 全十區 W8 QAT；若十區不能追回，正式主線回到已 green 的九區 W8。
3. 跑 nine-region-quality mixed-bit paired QAT，判定 W4/W5 是否能從 recover 回到雙 green。
4. 對 SD4 Detect head 建立 Fixed-SD4 與 learned-scale SD4 的可歸因 QAT 對照；不能把一般 joint recovery 全部宣稱為 LS-SD4 收益。
5. 補 regional Hardswish 與 qSiLU 的同一小型 coupling matrix。Q3 只支持三個單區 Hardswish policy，不能外推成 uniform 或多區 winner。
6. 只讓通過 search gate 的少數 finalists 進多 seed、formal validation 與 deployment export；final 選定前不做最終硬體特化。
7. 補 bias／padding correction、Add／Concat requant、accumulator saturation 與 bit packing 的 native integer／target hardware 驗證。沒有原生 W5/W6/W7 kernel前，不宣稱 GPU speedup。

## 證據

- V17 dual：`3e58104be08c6a6e89cacd274867bf01d1c19e6b421d0fdc32d1f1ce73ff7c10`
- W7／W6／W5／W4 dual：`8709384a...`／`cfc195a7...`／`3a341764...`／`1a5489d4...`
- mixed-bit dual：`24cb921d160b6e3031fdd2b2d063e539a012ab652f579142c372b597f9e3678f`
- special isolated dual：`bdd9cd2169e4f1827fdcf71ee317ed5c63cfe5dfd24baba46cb111fa193feade`
- coupled source／dual：`68fd8b3a31844390681e0b1692354c572cfc574d349fdae2bca7062db71dc28e`／`db05395178302647249911773a6a33fc4828006651fac797831a320cc3a00b1b`
- QAT GPU smoke v2：`3ad03a9c282b47d3f20ec3ba1b35fa765b6bba99452a3eba27e9db6ee3c2f551`

本報告中的 validation 都是 search validation，不是 formal validation；尚未宣稱 final winner 或硬體實測效能。
