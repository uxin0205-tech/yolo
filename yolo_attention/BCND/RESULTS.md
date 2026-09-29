# BCND v2/v3 results

目前狀態：v2 與 v3 全部完成，queue worker 已退出。

| Run | Range | Levels | Step | Profiler arithmetic | 含 bias 保守值 | Memory proxy | mAP50-95 | Overflow | Last bucket | Max distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BDCN-V2-LEARN | 0–8 | 64 | 0.126984 | 113,030,400 | 115,590,400 | 14,387,200 | 0.483281 | 0.205165 | 0.210540 | 34.6934 |
| BDCN-V2-R1 | 0–8 | 64 | 0.126984 | 113,008,000 | 115,568,000 | 14,387,200 | 0.483500 | 0.205146 | 0.210519 | 34.6934 |
| BDCN-V3-FIXED | 0–8 | 64 | 0.126984 | 113,030,400 | 115,590,400 | 14,387,200 | 0.506566 | 0.197186 | 0.202423 | 33.8770 |
| BDCN-V3-LEARN | 0–8 | 64 | 0.126984 | 113,030,400 | 115,590,400 | 14,387,200 | 0.506469 | 0.197678 | 0.202929 | 33.3770 |
| BDCN-V3-R1 | 0–8 | 64 | 0.126984 | 113,008,000 | 115,568,000 | 14,387,200 | 0.506562 | 0.197674 | 0.202925 | 33.3770 |

數值只可由各 run 的 `metrics/queue-result.json` 填入，不得用估計值替代。
V3-FIXED 比 V2-LEARN 高 0.023285 mAP，已證明 v2 的主要退化來自 unconstrained
codebook learning drift，而不是 64-level fixed exponential discretization 本身。
V3-LEARN 比 V3-FIXED 低 0.000097，沒有顯示 codebook training 的收益；V3-R1
與 V3-FIXED 只差 0.000005，顯示 reciprocal LUT 幾乎無損。完整分析見
[`reports/BDCN_V3_REPORT.md`](../reports/BDCN_V3_REPORT.md)。
