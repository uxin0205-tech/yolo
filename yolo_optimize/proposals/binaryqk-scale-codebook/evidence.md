# BinaryQK scale／codebook：證據與數值索引

本檔只整理來源、可下的結論與證據邊界；完整推導與外部一手來源集中在
[2026-09-03 研究報告](<../../docs/research/2026-09-03-binaryqk-scale-codebook-hardware.md>)，不在本
資料夾複製另一份研究全文。

## 一、本地可直接量到的證據

| 主張 | 類型 | 來源 | 可下的結論 |
|---|---|---|---|
| V1-DYN `0.507457458`、V1-P2 `0.506951921` | 單任務 YOLO26 evaluation | [comparison.csv](<../../../yolo_attention/reports/comparison.csv>) | global dynamic只比 PoT高 `0.000505537` |
| B26-FP `0.517997835` | 同一 comparison table | [comparison.csv](<../../../yolo_attention/reports/comparison.csv>) | FP–PoT gap為 `0.011045914` |
| PoT依 tie-band獲選 | 本地決策 | [REPORT.md](<../../../yolo_attention/reports/REPORT.md>) | 選 PoT是複雜度決策，不代表它的 mAP最高 |
| Full35採 Hadamard＋power-of-two | 正式設定 | [bittrue-pwl-final.yaml](<../../../yolo_combine/final/full35/source_bundle/configs/attention/bittrue-pwl-final.yaml>) | 正式 inference目標是 fixed PoT two-basis |
| global magnitude與 fixed coefficient流程 | 正式程式 | [binary_basis.py](<../../../yolo_combine/final/full35/source_bundle/code/yolo_attention/binary_basis.py>) | 現有 dynamic粒度是 image/head/basis global；不是 per-token |
| `N=400,H=4,D=32`與 two-site profiler | 正式程式 | [profiling.py](<../../../yolo_combine/final/full35/source_bundle/code/yolo_attention/profiling.py>) | 可建立本方向 operation／storage上界 |

計算：

~~~text
dynamic - PoT = 0.507457458 - 0.506951921
              = 0.000505537

FP - PoT      = 0.517997835 - 0.506951921
              = 0.011045914

gap closed    = 0.000505537 / 0.011045914 × 100%
              = 4.5767%
~~~

因此只把 global coefficient由 fixed改回每-image dynamic，不足以解釋或關閉主要精度缺口。

## 二、本地診斷性證據

兩張 COCO image的 CPU mechanism probe比較 global dynamic與 per-token dynamic：

| site | global top-10 | per-token top-10 | global KL | per-token KL |
|---|---:|---:|---:|---:|
| `model.10.m.0.attn` | 0.4206 | 0.5119 | 0.5431 | 0.4651 |
| `model.22.m.0.1.attn` | 0.4962 | 0.5952 | 0.8810 | 0.6188 |

這只支持「token magnitude值得作條件式 fidelity probe」，不代表：

- A8已提高 detection mAP。
- A8一定比正式 fixed-PoT A-FINAL好；probe control是 global dynamic。
- 兩張 image可代表完整 validation。
- runtime scale成本可接受。

probe背景與 P1優先順序見
[BinaryQK 精度恢復第一手證據報告](<../../docs/research/2026-09-02-binaryqk-accuracy-recovery-primary-sources.md>)。

## 三、由正式形狀推導的上界

令 `S=2, B=2, H=4, N=400, D=32`：

~~~text
P = S·H·N²         = 1,280,000 final token pairs
E = S·B·H·N²       = 2,560,000 basis terms
T = S·B·H·2·N      = 12,800 token-scale slots/image
V = T·D            = 409,600 Q/K values scanned/image
~~~

| 推導量 | 算式 | 結果 |
|---|---|---:|
| C0 fixed slots | `S·B·H` | 16 |
| B4 fixed slots | `S·B·H·4` | 64 |
| B4 partial terms上界 | `E·4` | 10,240,000 |
| A8 magnitude abs | `T·D` | 409,600 |
| A8 reduction adds | `T·(D-1)` | 396,800 |
| A8 3-bit index storage | `T·3/8` | 4,800 B |
| A8 variable shifts上界 | `E` | 2,560,000 |

這些值是本專案公式的 analytical上界，不是 profiler量到的 instruction、cycle、latency、能耗或
FPGA資源。只有 target kernel trace／microbenchmark可以回答實際硬體成本。

## 四、來源層級與不可外推事項

| 層級 | 本方向用途 | 不能外推成 |
|---|---|---|
| 本地正式 metrics／config／code | 建立現況、lineage與準確算術 | 新候選一定有效 |
| 本地兩圖 CPU probe | 找出可測機理與方向 | 完整 detection mAP |
| analytical operation count | 預先暴露成本項目 | 真實 latency或能耗 |
| BinaryAttention／LQ-Nets／DeepShift／PTX一手資料 | 支持 scaled-binary、PoT與 bit primitive可行性 | 本 YOLO架構的增益保證 |

外部來源與逐項引用見[完整研究報告](<../../docs/research/2026-09-03-binaryqk-scale-codebook-hardware.md>)。

## 五、目前仍缺的證據

- C0／B4／A8同 tensors的 machine-readable cached replay。
- B4 4-way partial-popcount target-kernel parity與 p50/p95 latency。
- A8 reduction／selector／index／epilogue全包含的 target profile。
- B4或A8完整 detection validation與 matched QAT。
- 三個 paired seeds及關鍵類別沒有退化的證據。

在這些證據完成前，本方向狀態維持 `proposed`，且不能宣稱回補約 `0.011`。

返回[方向說明](<README.md>)。
