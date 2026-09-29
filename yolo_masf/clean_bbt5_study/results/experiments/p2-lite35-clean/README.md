# P2-Lite35-Clean

在 P2 只保留 DW3、DW5 與兩層 1x1 residual fusion。

- Comparison tier：strict_fair
- Family：P2
- Schedule：direct100
- Seeds：42、43

| Split | mAP50–95 mean ± std | Ball AP | Bat AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.45127 ± 0.01204 | 0.21054 ± 0.01734 | 0.69200 ± 0.00674 | 0.21872 ± 0.02605 | 0.74697 ± 0.00943 | 0.89633 ± 0.02129 |
| test | 0.57019 ± 0.02350 | 0.52100 ± 0.03750 | 0.61937 ± 0.00949 | 0.36517 ± 0.00187 | 0.58767 ± 0.01612 | 0.65685 ± 0.03856 |

逐 seed 證據：

- artifacts/clean-bbt5-ablation/evaluation/val/p2-lite35-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/val/p2-lite35-clean-seed43/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-lite35-clean-seed42/
- artifacts/clean-bbt5-ablation/evaluation/test/p2-lite35-clean-seed43/
