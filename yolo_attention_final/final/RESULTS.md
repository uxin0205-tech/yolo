# YOLO26m PWL Attention 最終結果

## 結論

19 個 LR sweep／staged-recovery queue jobs 全部成功。正式 winner 仍是 zero-train Bit-True
`pwl-final-best.pt`；單一 seed 的最高實測 checkpoint 是 block x1
best-observed research export（不包含在可攜交付包；SHA-256 見 `results.json`），
mAP50-95 `0.5069387`。它比同輪 audited parent 高 `0.0001944`，
未達 `0.001` 穩健門檻，也沒有三個 seeds 的 mean，因此只能稱為 best-observed。

## 正式 Bit-True COCO2017 val 結果

| 候選 | mAP50-95 | mAP50 | mAP75 | 相對 audited parent | 結果 |
|---|---:|---:|---:|---:|---|
| Audited parent | 0.506744 | 0.677201 | 0.555110 | 0 | 對照組 |
| Block x1 | **0.506939** | **0.677632** | 0.555055 | +0.000194 | 最高觀測值 |
| Block x2 | 0.506434 | 0.676635 | 0.554547 | -0.000310 | 未選 |
| Block x4 | 0.505749 | 0.675902 | 0.553549 | -0.000995 | 未選 |
| Neck | 0.506276 | 0.676653 | 0.554492 | -0.000469 | Phase gate 通過；未獲全域選擇 |
| Backbone-last | 0.506237 | 0.676962 | 0.553944 | -0.000507 | Phase gate 通過；未獲全域選擇 |
| Full model | 0.506011 | 0.676664 | 0.553673 | -0.000733 | Phase gate 通過；未獲全域選擇 |

Phase gate 的含義是「child 相對直接 parent 下降不超過 0.001」，不是宣告 child 優於 parent。
最終 selector 比較所有候選後選回 block x1。B26-FP baseline 是 `0.517998`，best-observed
仍低 `0.0110593`。

## 方法與實跑設定

- Model：官方 YOLO26m、scale m、80 classes；兩個指定 Attention site 同時存在。
- Dataset：COCO2017 train 118,287 images；val 5,000 images；imgsz 640、batch 16、seed 0。
- Trainer：AdamW、AMP、deterministic、constant LR、weight decay `5e-4`、warmup 完全關閉。
- Block sweep：Attention LR `5e-6`／`1e-5`／`2e-5`，相鄰 block 是其五分之一；patience 3。
- Recovery：Attention `5e-6`、相鄰 block `1e-6`、Neck/Detect `5e-7`、Backbone `1e-7`。
- 實跑 epochs：block x1/x2/x4 為 5/7/4；Neck/Backbone/Full 為 6/7/8，皆由 early stopping 結束。
- 訓練使用可微分 Float-PWL；上表全部是另行 materialize 後的實際 Bit-True checkpoint 完整驗證。
- 非 Attention BN running statistics 在 recovery 中鎖定；Bit-True reference 沒有偷偷加入 STE。

## 來源與環境記錄

- Project revision：`3e4c5dc983ce5fe52e64e45e277883572b0648fe`
- Ultralytics：8.4.90，source commit `ea920761b1c6d531c2231d0033714301690cf67d`
- Python 3.12.3；PyTorch 2.11.0+cu128；CUDA 12.8；NVIDIA GeForce RTX 5090。
- Dataset YAML SHA-256：`e6556e609f623fabc7d076262b14d7ae953a9301b9ea3db242ff01437af4ffa0`
- Retained parent SHA-256：`9e2ca0d93793785f0f7e514d876580204070bac57b4163d7c5f0c5ca352b9c3f`
- Queue state：`artifacts/lr-sweep-queue` revision 95，19/19 succeeded。

`metrics.csv` 提供精簡比較，`results.json` 保存正式 winner 與 best-observed provenance。
完整 per-class AP、training curves、phase gates 與歷史 queue events 保留在原研究 repository，
不複製到可攜交付包；交付包只保留重跑必要的 Phase-B parent record。

## 限制與未完成項目

- 依使用者指示只完成 seed 0，沒有執行 seed 1/2，因此沒有 three-seed mean/std。
- 指標是 Ultralytics internal COCO2017 metric；本輪未安裝／使用 canonical COCO API gate。
- PWL numerator 是 Bit-True Q8.8/UQ1.15 reference，但 denominator 仍是 exact float。
- P/PV 與 row-sum 欄位在這批正式 detector evaluation JSON 中是 `null`；不得解讀為硬體量測已完成。
- 本輪沒有 KD、P/V/projection quantization、HLS、上板或 latency measurement。
